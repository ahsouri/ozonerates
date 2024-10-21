from scipy.spatial import Delaunay
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.spatial import cKDTree
from ozonerates.interpolator import _interpolosis
from ozonerates.config import param_input
from netCDF4 import Dataset
import os
from numpy import dtype
import numpy as np


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
        if ctm_freq=='daily':
           # find the closest day
           closest_index = np.argmin(np.abs(time_sat - time_ctm))
           # find the closest hour (this only works for 3-hourly frequency)
           closest_index_day = int(np.floor(closest_index/8.0))
           closest_index_hour = int(closest_index % 8)
        # if the frequency of the ctm data is monthly, we will focus only on hours
        if ctm_freq=='monthly':
           # find the closest hour only
            closest_index = np.argmin(
                np.abs(time_sat_hour_only - time_ctm_hour_only))
            # find the closest hour
            closest_index_hour = int(closest_index)
            closest_index_day = int(0)

        print("The closest GMI file used for the L2 at " + str(L2_granule.time) +
              " is at " + str(time_ctm_datetype[closest_index_day][closest_index_hour]))

        #ctm_mid_pressure = ctmdata[closest_index_day].pressure_mid[closest_index_hour, :, :, :].squeeze(
        #)
        ctm_no2_profile_factor = ctmdata[closest_index_day].gas_profile_no2[closest_index_hour, :, :].squeeze(
        )
        ctm_hcho_profile_factor = ctmdata[closest_index_day].gas_profile_hcho[closest_index_hour, :, :].squeeze(
        )
        #ctm_mid_T = ctmdata[closest_index_day].tempeature_mid[closest_index_hour, :, :, :].squeeze(
        #)
        #ctm_height = ctmdata[closest_index_day].height_mid[closest_index_hour, :, :, :].squeeze(
        #)
        ctm_O3col = ctmdata[closest_index_day].O3col[closest_index_hour, :, :].squeeze(
        )
        ctm_PBLH = ctmdata[closest_index_day].PBLH[closest_index_hour, :, :].squeeze(
        )
        ctm_H2O = ctmdata[closest_index_day].H2O[closest_index_hour, :, :].squeeze(
        )

        #ctm_mid_pressure_new = np.zeros((np.shape(ctm_mid_pressure)[0],
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
        #for z in range(0, np.shape(ctm_mid_pressure)[0]):
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

    #data6 = ncfile.createVariable(
    #    'pressure_mid', dtype('float32').char, ('z', 'x', 'y'))
    #data6[:, :, :] = data.pressure_mid

    #data7 = ncfile.createVariable(
    #    'temperature_mid', dtype('float32').char, ('z', 'x', 'y'))
    #data7[:, :, :] = data.tempeature_mid

    #data8 = ncfile.createVariable(
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

    #data3 = ncfile.createVariable(
    #    'time', 'S1', ('t'))
    #data3 = data.time.strftime("%Y-%m-%d %H:%M:%S")

    data4 = ncfile.createVariable(
        'vcd_no2', dtype('float32').char, ('t','x', 'y'))
    data4[:, :] = data.vcd_no2

    data5 = ncfile.createVariable(
        'vcd_no2_factor', dtype('float32').char, ('t', 'x', 'y'))
    data5[:, :] = data.vcd_no2_factor

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
