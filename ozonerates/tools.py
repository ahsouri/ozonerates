from scipy.spatial import Delaunay
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.spatial import cKDTree
from ozonerates.interpolator import _interpolosis
from ozonerates.config import param_input
from ozonerates.writer import ozonerate_netcdf_writer
from netCDF4 import Dataset
import os
from datetime import datetime, timezone
from numpy import dtype
import numpy as np
from scipy.io import loadmat
import warnings
from cftime import date2num
from time import asctime, gmtime, strftime

warnings.filterwarnings('ignore')


def log_message(msg, error=False):
    '''Print message with time stamp
    ARGS:
        msg(string): Message to be printed
        error(bool,optional): If true raise Error to calling program
    '''
    if error:
        m = '{0}: ERROR!!! {1}'.format(asctime(), msg)
        exit(m)
    else:
        m = '{0}: {1}'.format(asctime(), msg)
        print(m)


def ctmpost(satdata, ctmdata, ctm_freq):

    print('Mapping CTM data into Sat grid...')
    # list the time in ctm_data
    time_ctm = []
    time_ctm_hour_only = []
    time_ctm_datetype = []
    for ctm_granule in ctmdata:
        time_temp = ctm_granule.time
        for n in range(len(time_temp)):
            time_temp1 = time_temp[n].year*10000 + time_temp[n].month*100 +\
                time_temp[n].day + time_temp[n].hour/24.0 + \
                time_temp[n].minute/60.0/24.0 + time_temp[n].second/3600.0/24.0
            time_temp2 = time_temp[n].hour/24.0 + \
                time_temp[n].minute/60.0/24.0 + time_temp[n].second/3600.0/24.0
            time_ctm_hour_only.append(time_temp2)
            time_ctm.append(time_temp1)
        time_ctm_datetype.append(ctm_granule.time)
    time_ctm = np.array(time_ctm)
    time_ctm_hour_only = np.array(time_ctm_hour_only)
    # define the triangulation
    points = np.zeros((np.size(ctmdata[0].latitude), 2))
    points[:, 0] = ctmdata[0].longitude.flatten()
    points[:, 1] = ctmdata[0].latitude.flatten()
    tri = Delaunay(points)
    # loop over the satellite list
    counter = 0
    params = []
    for L2_granule in satdata:
        if (L2_granule is None):
            counter += 1
            continue
        time_sat_datetime = L2_granule.time
        time_sat = time_sat_datetime.year*10000 + time_sat_datetime.month*100 +\
            time_sat_datetime.day + time_sat_datetime.hour/24.0 + time_sat_datetime.minute / \
            60.0/24.0 + time_sat_datetime.second/3600.0/24.0
        time_sat_hour_only = time_sat_datetime.hour/24.0 + time_sat_datetime.minute / \
            60.0/24.0 + time_sat_datetime.second/3600.0/24.0
        # if the frequency of the ctm data is daily, we need to find the right time and day:
        if ctm_freq == 'daily':
            # find the closest day
            closest_index = np.argmin(np.abs(time_sat - time_ctm))
            # find the closest hour (this only works for 3-hourly frequency)
            closest_index_day = int(np.floor(closest_index/8.0))
            closest_index_hour = int(closest_index % 8)
        # if the frequency of the ctm data is monthly, we will focus only on hours
        if ctm_freq == 'monthly':
            # find the closest hour only
            closest_index = np.argmin(
                np.abs(time_sat_hour_only - time_ctm_hour_only))
            # find the closest hour
            closest_index_hour = int(closest_index)
            closest_index_day = int(0)

        print("The closest GMI file used for the L2 at " + str(L2_granule.time) +
              " is at " + str(time_ctm_datetype[closest_index_day][closest_index_hour]))

        # ctm_mid_pressure = ctmdata[closest_index_day].pressure_mid[closest_index_hour, :, :, :].squeeze(
        # )
        ctm_no2_profile_factor = ctmdata[closest_index_day].gas_profile_no2[closest_index_hour, :, :].squeeze(
        )
        ctm_hcho_profile_factor = ctmdata[closest_index_day].gas_profile_hcho[closest_index_hour, :, :].squeeze(
        )
        # ctm_mid_T = ctmdata[closest_index_day].tempeature_mid[closest_index_hour, :, :, :].squeeze(
        # )
        # ctm_height = ctmdata[closest_index_day].height_mid[closest_index_hour, :, :, :].squeeze(
        # )
        ctm_O3col = ctmdata[closest_index_day].O3col[closest_index_hour, :, :].squeeze(
        )
        ctm_PBLH = ctmdata[closest_index_day].PBLH[closest_index_hour, :, :].squeeze(
        )
        ctm_H2O = ctmdata[closest_index_day].H2O[closest_index_hour, :, :].squeeze(
        )

        # ctm_mid_pressure_new = np.zeros((np.shape(ctm_mid_pressure)[0],
        #                                 np.shape(L2_granule.longitude_center)[0], np.shape(
        #                                     L2_granule.longitude_center)[1],
        #                                 ))*np.nan

        #ctm_mid_T_new = np.zeros_like(ctm_mid_pressure_new)*np.nan
        #ctm_height_new = np.zeros_like(ctm_mid_pressure_new)*np.nan
        ctm_O3col_new = np.zeros((np.shape(L2_granule.longitude_center)[0], np.shape(
            L2_granule.longitude_center)[1],
        ))*np.nan
        ctm_PBLH_new = np.zeros((np.shape(L2_granule.longitude_center)[0], np.shape(
            L2_granule.longitude_center)[1],
        ))*np.nan
        ctm_H2O_new = np.zeros((np.shape(L2_granule.longitude_center)[0], np.shape(
            L2_granule.longitude_center)[1],
        ))*np.nan
        ctm_no2_profile_f_new = np.zeros_like(ctm_PBLH_new)*np.nan
        ctm_hcho_profile_f_new = np.zeros_like(ctm_PBLH_new)*np.nan
        sat_coordinate = {}
        sat_coordinate["Longitude"] = L2_granule.longitude_center
        sat_coordinate["Latitude"] = L2_granule.latitude_center
        # calculate distance to remove too-far estimates
        tree = cKDTree(points)
        grid = np.zeros((2, np.shape(sat_coordinate["Longitude"])[
                        0], np.shape(sat_coordinate["Longitude"])[1]))
        grid[0, :, :] = sat_coordinate["Longitude"]
        grid[1, :, :] = sat_coordinate["Latitude"]
        xi = _ndim_coords_from_arrays(tuple(grid), ndim=points.shape[1])
        dists, _ = tree.query(xi)
        # for z in range(0, np.shape(ctm_mid_pressure)[0]):
        #    ctm_mid_pressure_new[z, :, :] = _interpolosis(tri, ctm_mid_pressure[z, :, :].squeeze(), sat_coordinate["Longitude"],
        #                                                  sat_coordinate["Latitude"], 1, dists, 0.2)*L2_granule.quality_flag
        #    ctm_mid_T_new[z, :, :] = _interpolosis(tri, ctm_mid_T[z, :, :].squeeze(), sat_coordinate["Longitude"],
        #                                           sat_coordinate["Latitude"], 1, dists, 0.2)*L2_granule.quality_flag
        #    ctm_height_new[z, :, :] = _interpolosis(tri, ctm_height[z, :, :].squeeze(), sat_coordinate["Longitude"],
        #                                            sat_coordinate["Latitude"], 1, dists, 0.2)*L2_granule.quality_flag
        ctm_O3col_new[:, :] = _interpolosis(tri, ctm_O3col, sat_coordinate["Longitude"],
                                            sat_coordinate["Latitude"], 1, dists, 0.2)*L2_granule.quality_flag
        ctm_PBLH_new[:, :] = _interpolosis(tri, ctm_PBLH, sat_coordinate["Longitude"],
                                           sat_coordinate["Latitude"], 1, dists, 0.2)*L2_granule.quality_flag
        ctm_H2O_new[:, :] = _interpolosis(tri, ctm_H2O, sat_coordinate["Longitude"],
                                          sat_coordinate["Latitude"], 1, dists, 0.2)*L2_granule.quality_flag
        ctm_no2_profile_f_new[:, :] = _interpolosis(tri, ctm_no2_profile_factor, sat_coordinate["Longitude"],
                                                    sat_coordinate["Latitude"], 1, dists, 0.2)*L2_granule.quality_flag
        ctm_hcho_profile_f_new[:, :] = _interpolosis(tri, ctm_hcho_profile_factor, sat_coordinate["Longitude"],
                                                     sat_coordinate["Latitude"], 1, dists, 0.2)*L2_granule.quality_flag

        param = param_input(L2_granule.longitude_center, L2_granule.latitude_center, L2_granule.time,
                            ctm_no2_profile_f_new, ctm_hcho_profile_f_new, ctm_O3col_new, ctm_H2O_new, [],
                            [], [], ctm_PBLH_new, L2_granule.vcd, L2_granule.uncertainty,
                            L2_granule.tropopause, L2_granule.surface_albedo, L2_granule.SZA, L2_granule.surface_alt)
        params.append(param)

    return params


def write_to_nc(data, output_file, output_folder='diag'):
    ''' 
    Write the final results to a netcdf
    ARGS:
        output_file (char): the name of file to be outputted
    '''
    # writing
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ncfile = Dataset(output_folder + '/' + output_file + '.nc', 'w')

    # create the x and y dimensions.
    ncfile.createDimension('x', np.shape(data.latitude)[0])
    ncfile.createDimension('y', np.shape(data.latitude)[1])
    ncfile.createDimension('z', np.shape(data.height_mid)[0])
    ncfile.createDimension('t', None)

    data1 = ncfile.createVariable(
        'latitude', dtype('float32').char, ('x', 'y'))
    data1[:, :] = data.latitude

    data2 = ncfile.createVariable(
        'longitude', dtype('float32').char, ('x', 'y'))
    data2[:, :] = data.longitude

    data3 = ncfile.createVariable(
        'time', 'S1', ('t'))
    data3 = data.time.strftime("%Y-%m-%d %H:%M:%S")

    data4 = ncfile.createVariable(
        'gas_pbl_factor_no2', dtype('float32').char, ('x', 'y'))
    data4[:, :] = data.gas_profile_no2

    data5 = ncfile.createVariable(
        'gas_pbl_factor_hcho', dtype('float32').char, ('x', 'y'))
    data5[:, :] = data.gas_profile_hcho

    # data6 = ncfile.createVariable(
    #    'pressure_mid', dtype('float32').char, ('z', 'x', 'y'))
    #data6[:, :, :] = data.pressure_mid

    # data7 = ncfile.createVariable(
    #    'temperature_mid', dtype('float32').char, ('z', 'x', 'y'))
    #data7[:, :, :] = data.tempeature_mid

    # data8 = ncfile.createVariable(
    #    'height_mid', dtype('float32').char, ('z', 'x', 'y'))
    #data8[:, :, :] = data.height_mid

    data9 = ncfile.createVariable(
        'O3col', dtype('float32').char, ('x', 'y'))
    data9[:, :] = data.O3col

    data10 = ncfile.createVariable(
        'PBLH', dtype('float32').char, ('x', 'y'))
    data10[:, :] = data.PBLH

    data11 = ncfile.createVariable(
        'tropopause', dtype('float32').char, ('x', 'y'))
    data11[:, :] = data.tropopause

    data12 = ncfile.createVariable(
        'surface_albedo', dtype('float32').char, ('x', 'y'))
    data12[:, :] = data.surface_albedo

    data13 = ncfile.createVariable(
        'SZA', dtype('float32').char, ('x', 'y'))
    data13[:, :] = data.SZA

    data14 = ncfile.createVariable(
        'VCD', dtype('float32').char, ('x', 'y'))
    data14[:, :] = data.vcd

    data15 = ncfile.createVariable(
        'VCD_err', dtype('float32').char, ('x', 'y'))
    data15[:, :] = data.vcd_err

    data16 = ncfile.createVariable(
        'surface_alt', dtype('float32').char, ('x', 'y'))
    data16[:, :] = data.surface_alt

    data17 = ncfile.createVariable(
        'H2O', dtype('float32').char, ('x', 'y'))
    data17[:, :] = data.h2o

    ncfile.close()


def write_to_nc_product(data, output_file, output_folder='diag'):

    # Define filename
    filename = output_file
    file_full_address = output_folder + '/' + filename
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Define dictionary of dimensions in the file
    dimensions_dict = {
        'latitude': np.shape(data.vcd_no2)[2],
        'longitude': np.shape(data.vcd_no2)[1],
        'time': len(data.time)
    }
    # Define the reference time (1980-01-06T00:00:00Z) as a datetime object
    reference_time = datetime(1980, 1, 6, 0, 0, 0)
    # Define dictionary of global attributes
    global_attributes = {
        'geospatial_lon_min': np.min(np.ndarray.flatten(data.longitude)),
        'geospatial_lon_max':  np.max(np.ndarray.flatten(data.longitude)),
        'geospatial_lat_min':  np.min(np.ndarray.flatten(data.latitude)),
        'geospatial_lat_max':   np.max(np.ndarray.flatten(data.latitude)),
        'history': '{}: file created'.format(strftime('%Y-%m-%dT%H:%M:%SZ', gmtime())),
        'product_type': 'PO3',
        'processing_level': '4',
        'processing_version': 1.0,
        'production_date_time': strftime('%Y-%m-%dT%H:%M:%SZ', gmtime()),
        'local_granule_id': filename,
        'version_id': 1,
        'shortname': 'OZONERATES_PO3',
        'ozonerates_software_version': 1,
        'time_reference': '1980-01-06T00:00:00Z',
        'time_coverage_start': datetime.strptime(str(min(data.time)), "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%SZ"),
        'time_coverage_end': datetime.strptime(str(max(data.time)), "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%SZ"),
        'time_coverage_start_since_epoch': (min(data.time) - reference_time).total_seconds(),
        'time_coverage_end_since_epoch': (max(data.time) - reference_time).total_seconds(),
        'project': "Long-term Maps of Satellite-Based Ozone Production Rates using OMI and TROPOMI HCHO and NO2 Observations via Empirical and Machine Learning Methods: Insights from NASA's Air Quality Campaigns",
        'summary': 'Global ozone production rates derived from satellite observations, ground remote sensing, and atmospheric models, Souri et al., 2025 (https://egusphere.copernicus.org/preprints/2024/egusphere-2024-1947/)',
        'spatial_coverage': 'Global',
        'source': 'OMI',
        'keywords': 'GEOGRAPHIC REGION>GLOBAL, EARTH SCIENCE>ATMOSPHERE>AIR QUALITY> TROPOSPHERIC OZONE, EARTH SCIENCE>ATMOSPHERE>ATMOSPHERIC CHEMISTRY>OXYGEN COMPOUNDS>ATMOSPHERIC OZONE',
        'funding': 'NASA Aura-ACMAP 2022 80NSSC23K1250',
        'title': 'Global Ozone Production Rates',
        'institution': 'NASA GSFC (614)/GETARII(MSU)/SAO',
        'creator': 'Amir H. Souri (PI), Gonzalo Gonzalez Abad (Co-I)',
        'creator_url': 'https://www.ozonerates.space',
        'Conventions': "CF-1.6, ACDD-1.3"
    }
    try:
        with ozonerate_netcdf_writer(file_full_address) as writer:
            log_message('writing the final product ...')
            # Create dimensions
            for name, length in dimensions_dict.items():
                writer.create_dimension(name, length)
            # Create global attributes
            writer.create_global_attributes(global_attributes)
            # Create latitude variable
            attributes = {
                'standard_name': 'latitude',
                'long_name': 'latitude',
                'comment': 'latitude at grid box center',
                'units': 'degrees_north',
                'valid_min': -90.0,
                'valid_max': 90.0
            }
            writer.create_variable('latitude', datatype=np.float32, dimensions=('longitude', 'latitude'),
                                   values=data.latitude, attributes=attributes,
                                   least_significant_digig=2)
            # Create longitude variable
            attributes = {
                'standard_name': 'longitude',
                'long_name': 'longitude',
                'comment': 'longitude at grid box center',
                'units': 'degrees_east',
                'valid_min': -180.0,
                'valid_max': 180.0
            }
            writer.create_variable('longitude', datatype=np.float32, dimensions=('longitude', 'latitude'),
                                   values=data.longitude, attributes=attributes,
                                   least_significant_digig=2)
            # Create time variable
            attributes = {
                'standard_name': 'time',
                'long_name': 'time',
                'comment': 'time at the middle of the time range considered in the calculation of the PO3 map',
                'comment': 'gregorian',
                'units': 'seconds since 1980-01-06T00:00:00Z'
            }
            seconds_since_reference = [int((time_val - reference_time).total_seconds()) for time_val in data.time]
            writer.create_variable('time', datatype=np.float64, dimensions=('time'),
                                   values=seconds_since_reference,
                                   attributes=attributes,
                                   least_significant_digig=1)
            # Create ozone production variable
            attributes = {
                'standard_name': 'PO3',
                'long_name': 'PO3',
                'comment': 'net ozone production rates within planetary boundary layer derived from satellite observations',
                'units': 'ppbv hr-1',
                'valid_min': -100.0,
                'valid_max':  100.0
            }
            writer.create_variable('PO3', datatype=np.float32, dimensions=('time', 'longitude', 'latitude'),
                                   values=data.PO3, attributes=attributes,
                                   least_significant_digig=2)
            # Create NO2 contribution to ozone production variable
            attributes = {
                'standard_name': 'PO3_NO2',
                'long_name': 'PO3_NO2',
                'comment': 'NO2 contributions to ozone production rates within planetary boundary layer derived from satellite observations',
                'units': 'ppbv hr-1',
                'valid_min': -100.0,
                'valid_max':  100.0
            }
            writer.create_variable('PO3_NO2', datatype=np.float32, dimensions=('time', 'longitude', 'latitude'),
                                   values=data.no2_vmr_contrib, attributes=attributes,
                                   least_significant_digig=2)
            # Create HCHO contribution to ozone production variable
            attributes = {
                'standard_name': 'PO3_HCHO',
                'long_name': 'PO3_HCHO',
                'comment': 'HCHO contributions to ozone production rates within planetary boundary layer derived from satellite observations',
                'units': 'ppbv hr-1',
                'valid_min': -100.0,
                'valid_max':  100.0
            }
            writer.create_variable('PO3_HCHO', datatype=np.float32, dimensions=('time', 'longitude', 'latitude'),
                                   values=data.hcho_vmr_contrib, attributes=attributes,
                                   least_significant_digig=2)
            '''
            # Create other contributions to ozone production variable
            attributes = {
                'standard_name': 'PO3_other',
                'long_name': 'PO3_other',
                'comment': 'other net contributions to ozone production rates within planetary boundary layer derived from satellite observations',
                'units': 'ppbv hr-1',
                'valid_min': -100.0,
                'valid_max':  100.0
            }
            writer.create_variable('PO3_other',datatype=np.float32,dimensions=(),
                                   values=np.random.random(test_shape)*100.0,attributes=attributes,
                                   least_significant_digig=2)
            '''
            # Create ozone production random error variable
            attributes = {
                'standard_name': 'PO3_error_rand',
                'long_name': 'PO3_error_rand',
                'comment': 'satellite random error contributions to ozone production rates error',
                'units': 'ppbv hr-1',
                'valid_min':  0.0,
                'valid_max':  100.0
            }
            writer.create_variable('PO3_error_rand', datatype=np.float32, dimensions=('time', 'longitude', 'latitude'),
                                   values=data.po3_err_rand, attributes=attributes,
                                   least_significant_digig=2)
            # Create ozone production systematic error variable
            attributes = {
                'standard_name': 'PO3_error_sys',
                'long_name': 'PO3_error_sys',
                'comment': 'model conversion + bias correction + ozone rate model RMSE error contributions to ozone production rate error',
                'units': 'ppbv hr-1',
                'valid_min':  0.0,
                'valid_max':  100.0
            }
            writer.create_variable('PO3_error_sys', datatype=np.float32, dimensions=('time', 'longitude', 'latitude'),
                                   values=data.po3_err_sys, attributes=attributes,
                                   least_significant_digig=2)
            # Create NO2 VCD variable
            attributes = {
                'standard_name': 'NO2_VCD',
                'long_name': 'NO2_VCD',
                'comment': 'bias corrected nitrogen dioxide satellite vertical column density',
                'units': 'molecules cm-2',
                'valid_min': -1e18,
                'valid_max':  1e18
            }
            writer.create_variable('NO2_VCD', datatype=np.float32, dimensions=('time', 'longitude', 'latitude'),
                                   values=data.vcd_no2*1e15, attributes=attributes,
                                   least_significant_digig=2)
            # Create HCHO VCD variable
            attributes = {
                'standard_name': 'HCHO_VCD',
                'long_name': 'HCHO_VCD',
                'comment': 'bias corrected formaldehyde satellite vertical column density',
                'units': 'molecules cm-2',
                'valid_min': -1e18,
                'valid_max':  1e18
            }
            writer.create_variable('HCHO_VCD', datatype=np.float32, dimensions=('time', 'longitude', 'latitude'),
                                   values=data.vcd_hcho*1e15, attributes=attributes,
                                   least_significant_digig=2)
            # Create NO2 ppbv within PBL variable
            attributes = {
                'standard_name': 'NO2_ppbv',
                'long_name': 'NO2_ppbv',
                'comment': 'nitrogen dioxide volume mixing ratio in the planetary boundary layer derived from satellite vertical column density and MINDS profiles',
                'units': 'ppbv',
                'valid_min': 0.0,
                'valid_max': 100.0
            }
            writer.create_variable('NO2_ppbv', datatype=np.float32, dimensions=('time', 'longitude', 'latitude'),
                                   values=data.no2_vmr, attributes=attributes,
                                   least_significant_digig=2)
            # Create HCHO ppbv within PBL variable
            attributes = {
                'standard_name': 'HCHO_ppbv',
                'long_name': 'HCHO_ppbv',
                'comment': 'formaldehyde volume mixing ratio in the planetary boundary layer derived from satellite vertical column density and MINDS profiles',
                'units': 'ppbv',
                'valid_min': 0.0,
                'valid_max': 100.0
            }
            writer.create_variable('HCHO_ppbv', datatype=np.float32, dimensions=('time', 'longitude', 'latitude'),
                                   values=data.hcho_vmr, attributes=attributes,
                                   least_significant_digig=2)

            # Create H2O concentration within PBL variable
            attributes = {
                'standard_name': 'H2O',
                'long_name': 'H2O',
                'comment': 'water vapor concentrations in the planetary boundary layer',
                'units': 'molecules m-3',
                'valid_min': 0.0,
                'valid_max': 1e20
            }
            writer.create_variable('H2O',datatype=np.float32,dimensions=('time', 'longitude', 'latitude'),
                                   values=data.H2O*1e18,attributes=attributes,
                                   least_significant_digig=2)

            # Create JNO2 variable
            attributes = {
                'standard_name': 'JNO2',
                'long_name': 'JNO2',
                'comment': 'photolysis rates for jNO2 (NO2+hv)',
                'units': '1/s',
                'valid_min': 0.0,
                'valid_max': 1.0
            }
            writer.create_variable('JNO2', datatype=np.float32, dimensions=('time', 'longitude', 'latitude'),
                                   values=data.jno2*1e-3, attributes=attributes,
                                   least_significant_digig=8)
            # Create JO1D variable
            attributes = {
                'standard_name': 'JO1D',
                'long_name': 'JO1D',
                'comment': 'photolysis rates for jO1D (O3+hv)',
                'units': '1/s',
                'valid_min': 0.0,
                'valid_max': 1.0
            }
            writer.create_variable('JO1D', datatype=np.float32, dimensions=('time', 'longitude', 'latitude'),
                                   values=data.jo1d*1e-6, attributes=attributes,
                                   least_significant_digig=8)
            '''
            # Create SZA variable
            attributes = {
                'standard_name': 'SZA',
                'long_name': 'solaz_zenith_angle',
                'comment': 'solar zenith angle',
                'units': 'degrees',
                'valid_min':  90,
                'valid_max':   0
            }
            writer.create_variable('SZA',datatype=np.float32,dimensions=('longitude','latitude','time'),
                                   values=np.random.random(test_shape)*90,attributes=attributes,
                                   least_significant_digig=2)
            # Create surface albedo variable
            attributes = {
                'standard_name': 'surface_albedo',
                'long_name': 'surface_albedo',
                'comment': 'surface_albedo',
                'units': 'unitless',
                'valid_min': 1,
                'valid_max': 0
            }
            writer.create_variable('surface_albedo',datatype=np.float32,dimensions=('longitude','latitude','time'),
                                   values=np.random.random(test_shape),attributes=attributes,
                                   least_significant_digig=2)
            '''

    except Exception as e:
        log_message(e, error=True)


def write_to_nc_product_old(data, output_file, output_folder='diag'):
    ''' 
    Write the final results to a netcdf
    ARGS:
        output_file (char): the name of file to be outputted
    '''
    # writing
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ncfile = Dataset(output_folder + '/' + output_file + '.nc', 'w')

    # create the x and y dimensions.
    ncfile.createDimension('x', np.shape(data.latitude)[0])
    ncfile.createDimension('y', np.shape(data.latitude)[1])
    ncfile.createDimension('t', np.shape(data.vcd_no2)[0])

    data1 = ncfile.createVariable(
        'latitude', dtype('float32').char, ('x', 'y'))
    data1[:, :] = data.latitude

    data2 = ncfile.createVariable(
        'longitude', dtype('float32').char, ('x', 'y'))
    data2[:, :] = data.longitude

    # data3 = ncfile.createVariable(
    #    'time', 'S1', ('t'))
    #data3 = data.time.strftime("%Y-%m-%d %H:%M:%S")

    data4 = ncfile.createVariable(
        'vcd_no2', dtype('float32').char, ('t', 'x', 'y'))
    data4[:, :, :] = data.vcd_no2

    data5 = ncfile.createVariable(
        'vcd_no2_factor', dtype('float32').char, ('t', 'x', 'y'))
    data5[:, :, :] = data.vcd_no2_factor

    data6 = ncfile.createVariable(
        'vcd_hcho', dtype('float32').char, ('t', 'x', 'y'))
    data6[:, :, :] = data.vcd_hcho

    data7 = ncfile.createVariable(
        'PO3', dtype('float32').char, ('t', 'x', 'y'))
    data7[:, :, :] = data.PO3

    data8 = ncfile.createVariable(
        'FNR', dtype('float32').char, ('t', 'x', 'y'))
    data8[:, :, :] = data.FNR

    data9 = ncfile.createVariable(
        'hcho_vmr', dtype('float32').char, ('t', 'x', 'y'))
    data9[:, :, :] = data.hcho_vmr

    data10 = ncfile.createVariable(
        'no2_vmr', dtype('float32').char, ('t', 'x', 'y'))
    data10[:, :, :] = data.no2_vmr

    data11 = ncfile.createVariable(
        'vcd_hcho_factor', dtype('float32').char, ('t', 'x', 'y'))
    data11[:, :, :] = data.vcd_hcho_factor

    data12 = ncfile.createVariable(
        'jo1d', dtype('float32').char, ('t', 'x', 'y'))
    data12[:, :, :] = data.jo1d

    data13 = ncfile.createVariable(
        'jno2', dtype('float32').char, ('t', 'x', 'y'))
    data13[:, :, :] = data.jno2

    data14 = ncfile.createVariable(
        'hcho_vmr_contrib', dtype('float32').char, ('t', 'x', 'y'))
    data14[:, :, :] = data.hcho_vmr_contrib

    data15 = ncfile.createVariable(
        'no2_vmr_contrib', dtype('float32').char, ('t', 'x', 'y'))
    data15[:, :, :] = data.no2_vmr_contrib

    data16 = ncfile.createVariable(
        'jno2_contrib', dtype('float32').char, ('t', 'x', 'y'))
    data16[:, :, :] = data.jno2_contrib

    data17 = ncfile.createVariable(
        'jo1d_contrib', dtype('float32').char, ('t', 'x', 'y'))
    data17[:, :, :] = data.jo1d_contrib

    data18 = ncfile.createVariable(
        'PO3_err', dtype('float32').char, ('t', 'x', 'y'))
    data18[:, :, :] = data.po3_err

    data19 = ncfile.createVariable(
        'H2O', dtype('float32').char, ('t', 'x', 'y'))
    data19[:, :, :] = data.H2O

    data20 = ncfile.createVariable(
        'h2o_contrib', dtype('float32').char, ('t', 'x', 'y'))
    data20[:, :, :] = data.h2o_contrib

    ncfile.close()


def remove_non_numbers(lst):
    return [x for x in lst if isinstance(x, (int, float))]


def error_averager(error_X: np.array):
    error_Y = np.zeros((np.shape(error_X)[1], np.shape(error_X)[2]))*np.nan
    for i in range(0, np.shape(error_X)[1]):
        for j in range(0, np.shape(error_X)[2]):
            temp = []
            for k in range(0, np.shape(error_X)[0]):
                temp.append(error_X[k, i, j])
            temp = np.array(temp)
            temp[np.isinf(temp)] = np.nan
            temp2 = temp[~np.isnan(temp)]
            error_Y[i, j] = np.sum(temp2)/(np.size(temp2)**2)

    error_Y = np.sqrt(error_Y)
    return error_Y


def tropomi_albedo(tri, uv: bool, lat, lon, month: int):
    tropomi_data = loadmat('../data/tropomi_ler_uv_vis.mat')
    lat_tropomi = tropomi_data["lat"]
    lon_tropomi = tropomi_data["lon"]
    if uv:
        tropomi_surface_albedo = tropomi_data["c0_uv"][:, :,
                                                       month-1]+tropomi_data["minimum_LER_uv"][:, :, month-1]
    else:
        tropomi_surface_albedo = tropomi_data["c0_vis"][:, :,
                                                        month-1]+tropomi_data["minimum_LER_vis"][:, :, month-1]

    tropomi_surface_albedo = np.squeeze(tropomi_surface_albedo)
    # define the triangulation
    points = np.zeros((np.size(lon_tropomi), 2))
    points[:, 0] = lon_tropomi.flatten()
    points[:, 1] = lat_tropomi.flatten()
    #tri = Delaunay(points)
    # calculate distance to remove too-far estimates
    tree = cKDTree(points)
    grid = np.zeros((2, np.shape(lon)[
        0], np.shape(lon)[1]))
    grid[0, :, :] = lon
    grid[1, :, :] = lat
    xi = _ndim_coords_from_arrays(tuple(grid), ndim=points.shape[1])
    dists, _ = tree.query(xi)
    albedo = _interpolosis(tri, tropomi_surface_albedo, lon,
                           lat, 1, dists, 0.2)
    return albedo
