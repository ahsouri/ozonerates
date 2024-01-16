import numpy as np
from pathlib import Path
import datetime
import glob
from joblib import Parallel, delayed
from netCDF4 import Dataset
from ozonerates.config import satellite_amf, ctm_model
from ozonerates.interpolator import interpolator
import warnings
from scipy.io import savemat
import yaml

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _read_nc(filename, var):
    # reading nc files without a group
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    out = np.array(nc_fid.variables[var])
    nc_fid.close()
    return np.squeeze(out)


def _get_nc_attr(filename, var):
    # getting attributes
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    attr = {}
    for attrname in nc_fid.variables[var].ncattrs():
        attr[attrname] = getattr(nc_fid.variables[var], attrname)
    nc_fid.close()
    return attr


def _read_group_nc(filename, group, var):
    # reading nc files with a group structure
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    num_groups = len(group)
    if num_groups == 1:
        out = np.array(nc_fid.groups[group[0]].variables[var])
    elif num_groups == 2:
        out = np.array(nc_fid.groups[group[0]].groups[group[1]].variables[var])
    elif num_groups == 3:
        out = np.array(
            nc_fid.groups[group[0]].groups[group[1]].groups[group[2]].variables[var])
    elif num_groups == 4:
        out = np.array(
            nc_fid.groups[group[0]].groups[group[1]].groups[group[2]].groups[group[3]].variables[var])
    nc_fid.close()
    return np.squeeze(out)

def gmi_reader_wrapper(fname_met: str, fname_gas: str, fname_pbl: str) -> ctm_model:
     # a nested function to make the code in parallel
     print("Currently reading: " + fname_met.split('/')[-1])
     Mair = 28.97e-3
     g = 9.80665
     N_A = 6.02214076e23
     # read the data
     print("Currently reading: " + fname_met.split('/')[-1])
     ctmtype = "GMI"
     # read coordinates
     lon = _read_nc(fname_met, 'lon')
     lat = _read_nc(fname_met, 'lat')
     lons_grid, lats_grid = np.meshgrid(lon, lat)
     latitude = lats_grid
     longitude = lons_grid
     # read time
     time_min_delta = _read_nc(fname_met, 'time')
     time_attr = _get_nc_attr(fname_met, 'time')
     timebegin_date = str(time_attr["begin_date"])
     timebegin_time = str(time_attr["begin_time"])
     if len(timebegin_time) == 5:
         timebegin_time = "0" + timebegin_time
         timebegin_date = [int(timebegin_date[0:4]), int(
            timebegin_date[4:6]), int(timebegin_date[6:8])]
         timebegin_time = [int(timebegin_time[0:2]), int(
            timebegin_time[2:4]), int(timebegin_time[4:6])]
     time = []
     for t in range(0, np.size(time_min_delta)):
         time.append(datetime.datetime(timebegin_date[0], timebegin_date[1], timebegin_date[2],
                           timebegin_time[0], timebegin_time[1], timebegin_time[2]) +
                           datetime.timedelta(minutes=int(time_min_delta[t])))
     # read pressure, temperature, PBL and other met information
     delta_p = _read_nc(fname_met, 'DELP').astype('float32')/100.0
     delta_p = np.flip(delta_p, axis=1)  # from bottom to top
     pressure_mid = _read_nc(fname_met, 'PL').astype('float32')/100.0
     pressure_mid = np.flip(pressure_mid, axis=1)  # from bottom to top
     temperature_mid = _read_nc(fname_met, 'T').astype('float32')
     temperature_mid = np.flip(
         temperature_mid, axis=1)  # from bottom to top
     height_mid = _read_nc(fname_met, 'H')/1000.0
     height_mid = np.flip(height_mid, axis=1)  # from bottom to top
     PBL = _read_nc(fname_pbl, 'PBLTOP')/100.0
     PBL = PBL[2::3, :, :]  # 1-hourly to 3-hourly
     tropp = _read_nc(fname_pbl, 'TROPPB')/100.0
     # read ozone
     O3 = np.flip(_read_nc(
         fname_gas, 'O3'), axis=1)
     # integrate ozone (dobson unit)
     O3 = O3*delta_p/g/Mair*N_A*1e-4*100.0/2.69e16
     O3 = np.sum(O3, axis=1).squeeze()
     # read hcho profiles
     HCHO = np.flip(_read_nc(
         fname_gas, 'CH2O'), axis=1)
     # making a mask for the PBL region (it's 4D)
     mask_PBL = np.zeros_like(pressure_mid)
     for a in range(0, np.shape(mask_PBL)[0]):
         for b in range(0, np.shape(mask_PBL)[1]):
             mask_PBL[a, b, :, :] = pressure_mid[a, b, :,
                                  :].squeeze() >= PBL[a, :, :].squeeze()
     mask_PBL = np.multiply(mask_PBL, 1.0).squeeze()
     mask_PBL[mask_PBL != 1.0] = np.nan
     # calculate the conversion of total HCHO to surface mixing ratio in ppbv
     HCHO = np.nanmean(1e9*HCHO*mask_PBL, axis=1).squeeze() / \
         np.sum(HCHO*delta_p/g/Mair*N_A*1e-4*100.0*1e-15, axis=1).squeeze()
     # calculate no2 profiles
     NO2 = np.flip(_read_nc(
         fname_gas, 'NO2'), axis=1)
     # making a mask for the troposphere
     mask_trop = np.zeros_like(pressure_mid)
     for a in range(0, np.shape(mask_trop)[0]):
         for b in range(0, np.shape(mask_trop)[1]):
             mask_trop[a, b, :, :] = pressure_mid[a, b, :,
                                   :].squeeze() >= tropp[a, :, :].squeeze()
     mask_trop = np.multiply(mask_trop, 1.0).squeeze()
     mask_trop[mask_trop != 1.0] = np.nan
     # calculate the conversion of trop NO2 to surface mixing ratio in ppbv
     NO2 = np.nanmean(1e9*NO2*mask_PBL, axis=1).squeeze()/np.nansum(NO2 *
                                     mask_trop*delta_p/g/Mair*N_A*1e-4*100.0*1e-15, axis=1).squeeze()
     #subset the vertical grids to reduce memory usage
     pressure_mid = pressure_mid[:,0:24,:,:]
     temperature_mid = temperature_mid[:,0:24,:,:]
     height_mid = height_mid[:,0:24,:,:]
     # shape up the ctm class
     gmi_data = ctm_model(latitude, longitude, time, NO2.astype('float16'), HCHO.astype('float16'), O3.astype('float16'),
                                  pressure_mid.astype('float16'), temperature_mid.astype('float16'), height_mid.astype('float16'), PBL.astype('float16'), ctmtype)
     return gmi_data

def GMI_reader(product_dir: str, YYYYMM: str, num_job=1) -> list:
    '''
       GMI reader
       Inputs:
             product_dir [str]: the folder containing the GMI data
             YYYYMM [str]: the target month and year, e.g., 202005 (May 2020)
            num_obj [int]: number of jobs for parallel computation
       Output:
             gmi_fields [ctm_model]: a dataclass format (see config.py)
    '''
    # read meteorological and chemical fields
    tavg3_3d_met_files = sorted(
        glob.glob(product_dir + "/*tavg3_3d_met_Nv." + str(YYYYMM) + "*.nc4"))
    tavg3_3d_gas_files = sorted(
        glob.glob(product_dir + "/*tavg3_3d_tac_Nv." + str(YYYYMM) + "*.nc4"))
    tavg1_2d_pbl = sorted(
        glob.glob(product_dir + "/*tavg1_2d_slv_Nx." + str(YYYYMM) + "*.nc4"))
    if len(tavg3_3d_gas_files) != len(tavg3_3d_met_files):
        raise Exception(
            "the data are not consistent")
    # define gas profiles for saving
    outputs = Parallel(n_jobs=num_job)(delayed(gmi_reader_wrapper)(
        tavg3_3d_met_files[k], tavg3_3d_gas_files[k], tavg1_2d_pbl[k]) for k in range(len(tavg3_3d_met_files)))
    return outputs


def tropomi_reader_hcho(fname: str, ctm_models_coordinate=None, read_ak=True) -> satellite_amf:
    '''
       TROPOMI HCHO L2 reader
       Inputs:
             fname [str]: the name path of the L2 file
             ctm_models_coordinate [dict]: a dictionary containing ctm lat and lon
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal 
       Output:
             tropomi_hcho [satellite_amf]: a dataclass format (see config.py)
    '''
    # hcho reader
    print("Currently reading: " + fname.split('/')[-1])
    # read time
    time = _read_group_nc(fname, ['PRODUCT'], 'time') +\
        np.nanmean(np.array(_read_group_nc(
            fname, ['PRODUCT'], 'delta_time')), axis=1)/1000.0
    time = np.nanmean(time, axis=0)
    time = np.squeeze(time)
    time = datetime.datetime(
        2010, 1, 1) + datetime.timedelta(seconds=int(time))
    #print(datetime.datetime.strptime(str(tropomi_hcho.time),"%Y-%m-%d %H:%M:%S"))
    # read lat/lon at centers
    latitude_center = _read_group_nc(
        fname, ['PRODUCT'], 'latitude').astype('float32')
    longitude_center = _read_group_nc(
        fname, ['PRODUCT'], 'longitude').astype('float32')
    # read total amf
    amf_total = _read_group_nc(fname, ['PRODUCT', 'SUPPORT_DATA', 'DETAILED_RESULTS'],
                               'formaldehyde_tropospheric_air_mass_factor')
    # read total hcho
    vcd = _read_group_nc(fname, ['PRODUCT'],
                         'formaldehyde_tropospheric_vertical_column')
    scd = _read_group_nc(fname, ['PRODUCT'], 'formaldehyde_tropospheric_vertical_column') *\
        amf_total
    vcd = (vcd*6.02214*1e19*1e-15).astype('float16')
    scd = (scd*6.02214*1e19*1e-15).astype('float16')
    # read quality flag
    quality_flag = _read_group_nc(
        fname, ['PRODUCT'], 'qa_value').astype('float16')
    # read pressures for SWs
    tm5_a = _read_group_nc(
        fname, ['PRODUCT', 'SUPPORT_DATA', 'INPUT_DATA'], 'tm5_constant_a')/100.0
    tm5_b = _read_group_nc(
        fname, ['PRODUCT', 'SUPPORT_DATA', 'INPUT_DATA'], 'tm5_constant_b')
    ps = _read_group_nc(fname, [
                        'PRODUCT', 'SUPPORT_DATA', 'INPUT_DATA'], 'surface_pressure').astype('float32')/100.0
    surface_alt = _read_group_nc(fname, [
                        'PRODUCT', 'SUPPORT_DATA', 'INPUT_DATA'], 'surface_altitude').astype('float32')
    surface_albedo = _read_group_nc(fname, [
                        'PRODUCT', 'SUPPORT_DATA', 'INPUT_DATA'], 'surface_albedo').astype('float32')
    SZA = _read_group_nc(fname, [
                        'PRODUCT', 'SUPPORT_DATA', 'GEOLOCATIONS'], 'solar_zenith_angle').astype('float32')
    p_mid = np.zeros(
        (34, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float32')
    if read_ak == True:
        SWs = np.zeros(
            (34, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float16')
        AKs = _read_group_nc(fname, [
            'PRODUCT', 'SUPPORT_DATA', 'DETAILED_RESULTS'], 'averaging_kernel').astype('float16')
    else:
        SWs = np.empty((1))
    # for some reason, in the HCHO product, a and b values are the center instead of the edges (unlike NO2)
    for z in range(0, 34):
        p_mid[z, :, :] = (tm5_a[z]+tm5_b[z]*ps[:, :])
        if read_ak == True:
            SWs[z, :, :] = AKs[:, :, z]*amf_total
    # remove bad SWs
    SWs[np.where((np.isnan(SWs)) | (np.isinf(SWs)) |
                 (SWs > 100.0) | (SWs < 0.0))] = 0.0
    # read the precision
    uncertainty = _read_group_nc(fname, ['PRODUCT'],
                                 'formaldehyde_tropospheric_vertical_column_precision')
    uncertainty = (uncertainty*6.02214*1e19*1e-15).astype('float16')

    tropomi_hcho = satellite_amf(vcd, scd, time, np.empty((1)), latitude_center, longitude_center,
                                [], [], uncertainty, quality_flag, p_mid, SWs, [], [], [], surface_albedo, SZA, surface_alt)
    # interpolation
    if (ctm_models_coordinate is not None):
        print('Currently interpolating ...')
        grid_size = 0.10  # degree
        tropomi_hcho = interpolator(
            1, grid_size, tropomi_hcho, ctm_models_coordinate, flag_thresh=0.5)
    # return
    return tropomi_hcho


def tropomi_reader_no2(fname: str, trop: bool, ctm_models_coordinate=None, read_ak=False) -> satellite_amf:
    '''
       TROPOMI NO2 L2 reader
       Inputs:
             fname [str]: the name path of the L2 file
             trop [bool]: true for considering the tropospheric region only
             ctm_models_coordinate [dict]: a dictionary containing ctm lat and lon
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal 
       Output:
             tropomi_no2 [satellite_amf]: a dataclass format (see config.py)
    '''
    # say which file is being read
    print("Currently reading: " + fname.split('/')[-1])
    # read time
    time = _read_group_nc(fname, ['PRODUCT'], 'time') +\
        np.nanmean(np.array(_read_group_nc(
            fname, ['PRODUCT'], 'delta_time')), axis=0)/1000.0
    time = np.squeeze(time)
    time = datetime.datetime(
        2010, 1, 1) + datetime.timedelta(seconds=int(time))
    #print(datetime.datetime.strptime(str(tropomi_no2.time),"%Y-%m-%d %H:%M:%S"))
    # read lat/lon at centers
    latitude_center = _read_group_nc(
        fname, ['PRODUCT'], 'latitude').astype('float32')
    longitude_center = _read_group_nc(
        fname, ['PRODUCT'], 'longitude').astype('float32')
    # read total amf
    amf_total = _read_group_nc(fname, ['PRODUCT'], 'air_mass_factor_total')
    # read no2
    if trop == False:
        vcd = _read_group_nc(
            fname, ['PRODUCT', 'SUPPORT_DATA', 'DETAILED_RESULTS'], 'nitrogendioxide_total_column')
        scd = _read_group_nc(
            fname, ['PRODUCT', 'SUPPORT_DATA', 'DETAILED_RESULTS'], 'nitrogendioxide_slant_column_density')
        # read the precision
        uncertainty = _read_group_nc(fname, ['PRODUCT', 'SUPPORT_DATA', 'DETAILED_RESULTS'],
                                     'nitrogendioxide_total_column_precision')
    else:
        vcd = _read_group_nc(
            fname, ['PRODUCT'], 'nitrogendioxide_tropospheric_column')
        scd = vcd*_read_group_nc(
            fname, ['PRODUCT'], 'air_mass_factor_troposphere')
        # read the precision
        uncertainty = _read_group_nc(fname, ['PRODUCT'],
                                     'nitrogendioxide_tropospheric_column_precision')
    vcd = (vcd*6.02214*1e19*1e-15).astype('float16')
    scd = (scd*6.02214*1e19*1e-15).astype('float16')
    uncertainty = (uncertainty*6.02214*1e19*1e-15).astype('float16')
    # read quality flag
    quality_flag = _read_group_nc(
        fname, ['PRODUCT'], 'qa_value').astype('float16')
    # read pressures for SWs
    tm5_a = _read_group_nc(fname, ['PRODUCT'], 'tm5_constant_a')/100.0
    tm5_a = np.concatenate((tm5_a[:, 0], 0), axis=None)
    tm5_b = _read_group_nc(fname, ['PRODUCT'], 'tm5_constant_b')
    tm5_b = np.concatenate((tm5_b[:, 0], 0), axis=None)

    ps = _read_group_nc(fname, [
                        'PRODUCT', 'SUPPORT_DATA', 'INPUT_DATA'], 'surface_pressure').astype('float32')/100.0
    surface_alt = _read_group_nc(fname, [
                        'PRODUCT', 'SUPPORT_DATA', 'INPUT_DATA'], 'surface_altitude').astype('float32')
    surface_albedo = _read_group_nc(fname, [
                        'PRODUCT', 'SUPPORT_DATA', 'INPUT_DATA'], 'surface_albedo').astype('float32')
    SZA = _read_group_nc(fname, [
                        'PRODUCT', 'SUPPORT_DATA', 'GEOLOCATIONS'], 'solar_zenith_angle').astype('float32')
    p_mid = np.zeros(
        (34, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float16')
    if read_ak == True:
        SWs = np.zeros(
            (34, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float16')
        AKs = _read_group_nc(fname, ['PRODUCT'],
                             'averaging_kernel').astype('float16')
    else:
        SWs = np.empty((1))
    for z in range(0, 34):
        p_mid[z, :, :] = 0.5*(tm5_a[z]+tm5_b[z]*ps[:, :] +
                              tm5_a[z+1]+tm5_b[z+1]*ps[:, :])
        if read_ak == True:
            SWs[z, :, :] = AKs[:, :, z]*amf_total
    # remove bad SWs
    SWs[np.where((np.isnan(SWs)) | (np.isinf(SWs)) |
                 (SWs > 100.0) | (SWs < 0.0))] = 0.0
    # read the tropopause layer index
    if trop == True:
        trop_layer = _read_group_nc(
            fname, ['PRODUCT'], 'tm5_tropopause_layer_index')
        tropopause = np.zeros_like(trop_layer).astype('float16')
        for i in range(0, np.shape(trop_layer)[0]):
            for j in range(0, np.shape(trop_layer)[1]):
                if (trop_layer[i, j] > 0 and trop_layer[i, j] < 34):
                    tropopause[i, j] = p_mid[trop_layer[i, j], i, j]
                else:
                    tropopause[i, j] = np.nan
    else:
        tropopause = np.empty((1))
    tropomi_no2 = satellite_amf(vcd, scd, time, tropopause, latitude_center, longitude_center,
                                [], [], uncertainty, quality_flag, p_mid, SWs, [], [], [], surface_albedo, SZA, surface_alt)
    # interpolation
    if (ctm_models_coordinate is not None):
        print('Currently interpolating ...')
        grid_size = 0.10  # degree
        tropomi_no2 = interpolator(
            1, grid_size, tropomi_no2, ctm_models_coordinate, flag_thresh=0.75)
    # return
    return tropomi_no2


def omi_reader_no2(fname: str, trop: bool, ctm_models_coordinate=None, read_ak=False) -> satellite_amf:
    '''
       OMI NO2 L2 reader
       Inputs:
             fname [str]: the name path of the L2 file
             trop [bool]: true for considering the tropospheric region only
             ctm_models_coordinate [dict]: a dictionary containing ctm lat and lon
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal 
       Output:
             omi_no2 [satellite_amf]: a dataclass format (see config.py)
    '''
    # say which file is being read
    print("Currently reading: " + fname.split('/')[-1])
    # read time
    time = _read_group_nc(fname, ['GEOLOCATION_DATA'], 'Time')
    time = np.squeeze(np.nanmean(time))
    time = datetime.datetime(
        1993, 1, 1) + datetime.timedelta(seconds=int(time))
    #print(datetime.datetime.strptime(str(time),"%Y-%m-%d %H:%M:%S"))
    # read lat/lon at centers
    latitude_center = _read_group_nc(
        fname, ['GEOLOCATION_DATA'], 'Latitude').astype('float32')
    longitude_center = _read_group_nc(
        fname, ['GEOLOCATION_DATA'], 'Longitude').astype('float32')
    # read no2
    if trop == False:
        vcd = _read_group_nc(
            fname, ['SCIENCE_DATA'], 'ColumnAmountNO2')
        scd = _read_group_nc(fname, ['SCIENCE_DATA'], 'AmfTrop') *\
            _read_group_nc(fname, ['SCIENCE_DATA'], 'ColumnAmountNO2Trop') +\
            _read_group_nc(fname, ['SCIENCE_DATA'], 'AmfStrat') *\
            _read_group_nc(fname, ['SCIENCE_DATA'], 'ColumnAmountNO2Strat')
        # read the precision
        uncertainty = _read_group_nc(fname, ['SCIENCE_DATA'],
                                     'ColumnAmountNO2Std')
    else:
        vcd = _read_group_nc(
            fname, ['SCIENCE_DATA'], 'ColumnAmountNO2Trop')
        scd = _read_group_nc(fname, ['SCIENCE_DATA'], 'AmfTrop') *\
            _read_group_nc(fname, ['SCIENCE_DATA'], 'ColumnAmountNO2Trop')
        # read the precision
        uncertainty = _read_group_nc(fname, ['SCIENCE_DATA'],
                                     'ColumnAmountNO2TropStd')
    vcd = (vcd*1e-15).astype('float16')
    scd = (scd*1e-15).astype('float16')
    uncertainty = (uncertainty*1e-15).astype('float16')

    SZA = _read_group_nc(
        fname, ['GEOLOCATION_DATA'], 'SolarZenithAngle').astype('float32')
    surface_terrain = _read_group_nc(
        fname, ['ANCILLARY_DATA'], 'TerrainHeight').astype('float32')
    # read quality flag
    cf_fraction = quality_flag_temp = _read_group_nc(
        fname, ['ANCILLARY_DATA'], 'CloudFraction').astype('float16')
    cf_fraction_mask = cf_fraction < 0.3
    cf_fraction_mask = np.multiply(cf_fraction_mask, 1.0).squeeze()

    train_ref = quality_flag_temp = _read_group_nc(
        fname, ['ANCILLARY_DATA'], 'TerrainReflectivity').astype('float16')
    train_ref_mask = train_ref < 0.6
    train_ref_mask = np.multiply(train_ref_mask, 1.0).squeeze()

    quality_flag_temp = _read_group_nc(
        fname, ['SCIENCE_DATA'], 'VcdQualityFlags').astype('float16')
    quality_flag = np.zeros_like(quality_flag_temp)*-100.0
    for i in range(0, np.shape(quality_flag)[0]):
        for j in range(0, np.shape(quality_flag)[1]):
            flag = '{0:08b}'.format(int(quality_flag_temp[i, j]))
            if flag[-1] == '0':
                quality_flag[i, j] = 1.0
            if flag[-1] == '1':
                if flag[-2] == '0':
                    quality_flag[i, j] = 1.0
    quality_flag = quality_flag*cf_fraction_mask*train_ref_mask
    # remove edges because their footprint is large
    quality_flag[:,0:2]=-100.0
    quality_flag[:,-2::]=-100.0
    # read pressures for SWs
    ps = _read_group_nc(fname, ['GEOLOCATION_DATA'],
                        'ScatteringWeightPressure').astype('float16')
    p_mid = np.zeros(
        (35, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float16')
    if read_ak == True:
        SWs = _read_group_nc(fname, ['SCIENCE_DATA'],
                             'ScatteringWeight').astype('float16')
        SWs = SWs.transpose((2, 0, 1))
    else:
        SWs = np.empty((1))
    for z in range(0, 35):
        p_mid[z, :, :] = ps[z]
    # remove bad SWs
    SWs[np.where((np.isnan(SWs)) | (np.isinf(SWs)) |
                 (SWs > 100.0) | (SWs < 0.0))] = 0.0
    # read the tropopause pressure
    if trop == True:
        tropopause = _read_group_nc(
            fname, ['ANCILLARY_DATA'], 'TropopausePressure').astype('float16')
    else:
        tropopause = np.empty((1))
    # populate omi class
    omi_no2 = satellite_amf(vcd, scd, time, tropopause, latitude_center,
                            longitude_center, [], [], uncertainty, quality_flag, p_mid, SWs,
                            [], [], [], train_ref, SZA, surface_terrain)
    # interpolation
    if (ctm_models_coordinate is not None):
        print('Currently interpolating ...')
        grid_size = 0.25  # degree
        omi_no2 = interpolator(
            1, grid_size, omi_no2, ctm_models_coordinate, flag_thresh=0.0)  # bilinear mass-conserved interpolation
    # return
    return omi_no2


def omi_reader_hcho(fname: str, ctm_models_coordinate=None, read_ak=False) -> satellite_amf:
    '''
       OMI HCHO L2 reader
       Inputs:
             fname [str]: the name path of the L2 file
             ctm_models_coordinate [dict]: a dictionary containing ctm lat and lon
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal 
       Output:
             omi_hcho [satellite_amf]: a dataclass format (see config.py)
    '''
    # we add "try" because some files have format issues thus unreadable
    try:
        # say which file is being read
        print("Currently reading: " + fname.split('/')[-1])
        # read time
        time = _read_group_nc(fname, ['geolocation'], 'time')
        time = np.squeeze(np.nanmean(time))
        time = datetime.datetime(
            1993, 1, 1) + datetime.timedelta(seconds=int(time))
        # read lat/lon at centers
        latitude_center = _read_group_nc(
            fname, ['geolocation'], 'latitude').astype('float32')
        longitude_center = _read_group_nc(
            fname, ['geolocation'], 'longitude').astype('float32')
        # read hcho
        vcd = _read_group_nc(
            fname, ['key_science_data'], 'column_amount')
        scd = _read_group_nc(fname, ['support_data'], 'amf') *\
            _read_group_nc(fname, ['key_science_data'], 'column_amount')
        # read the precision
        uncertainty = _read_group_nc(fname, ['key_science_data'],
                                     'column_uncertainty')
        vcd = (vcd*1e-15).astype('float16')
        scd = (scd*1e-15).astype('float16')
        uncertainty = (uncertainty*1e-15).astype('float16')
        # read quality flag
        cf_fraction = _read_group_nc(
            fname, ['support_data'], 'cloud_fraction').astype('float16')
        cf_fraction_mask = cf_fraction < 0.4
        cf_fraction_mask = np.multiply(cf_fraction_mask, 1.0).squeeze()

        surface_albedo = _read_group_nc(
            fname, ['support_data'], 'albedo').astype('float16')
        SZA = _read_group_nc(
            fname, ['geolocation'], 'solar_zenith_angle').astype('float32')
        terrain_height = _read_group_nc(
            fname, ['geolocation'], 'terrain_height').astype('float32')
        quality_flag = _read_group_nc(
            fname, ['key_science_data'], 'main_data_quality_flag').astype('float16')
        quality_flag = quality_flag == 0.0
        quality_flag = np.multiply(quality_flag, 1.0).squeeze()

        quality_flag = quality_flag*cf_fraction_mask
        # remove edges because their footprint is large
        quality_flag[:,0:2]=-100.0
        quality_flag[:,-2::]=-100.0
        # read pressures for SWs
        ps = _read_group_nc(fname, ['support_data'],
                            'surface_pressure').astype('float16')
        a0 = np.array([0., 0.04804826, 6.593752, 13.1348, 19.61311, 26.09201, 32.57081, 38.98201, 45.33901, 51.69611, 58.05321, 64.36264, 70.62198, 78.83422, 89.09992, 99.36521, 109.1817, 118.9586, 128.6959, 142.91, 156.26, 169.609, 181.619,
                       193.097, 203.259, 212.15, 218.776, 223.898, 224.363, 216.865, 201.192, 176.93, 150.393, 127.837, 108.663, 92.36572, 78.51231, 56.38791, 40.17541, 28.36781, 19.7916, 9.292942, 4.076571, 1.65079, 0.6167791, 0.211349, 0.06600001, 0.01])
        b0 = np.array([1., 0.984952, 0.963406, 0.941865, 0.920387, 0.898908, 0.877429, 0.856018, 0.8346609, 0.8133039, 0.7919469, 0.7706375, 0.7493782, 0.721166, 0.6858999, 0.6506349, 0.6158184, 0.5810415, 0.5463042,
                       0.4945902, 0.4437402, 0.3928911, 0.3433811, 0.2944031, 0.2467411, 0.2003501, 0.1562241, 0.1136021, 0.06372006, 0.02801004, 0.006960025, 8.175413e-09, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        p_mid = np.zeros(
            (np.size(a0)-1, np.shape(vcd)[0], np.shape(vcd)[1])).astype('float16')
        if read_ak == True:
            SWs = _read_group_nc(fname, ['support_data'],
                                 'scattering_weights').astype('float16')
        else:
            SWs = np.empty((1))
        for z in range(0, np.size(a0)-1):
            p_mid[z, :, :] = 0.5*((a0[z] + b0[z]*ps) + (a0[z+1] + b0[z+1]*ps))
        # remove bad SWs
        SWs[np.where((np.isnan(SWs)) | (np.isinf(SWs)) |
                     (SWs > 100.0) | (SWs < 0.0))] = 0.0
        # no need to read tropopause for hCHO
        tropopause = np.empty((1))
        # populate omi class
        omi_hcho = satellite_amf(vcd, scd, time, tropopause, latitude_center,
                                 longitude_center, [], [], uncertainty, quality_flag, p_mid, SWs, [], [], [], surface_albedo, SZA, terrain_height)
        # interpolation
        if (ctm_models_coordinate is not None):
            print('Currently interpolating ...')
            grid_size = 0.25  # degree
            omi_hcho = interpolator(
                1, grid_size, omi_hcho, ctm_models_coordinate, flag_thresh=0.0)
        # return
        return omi_hcho
    except:
        return None


def tropomi_reader(product_dir: str, satellite_product_name: str, ctm_models_coordinate: dict, YYYYMM: str, trop: bool, read_ak=True, num_job=1):
    '''
        reading tropomi data
             product_dir [str]: the folder containing the tropomi data
             satellite_product_name [str]: so far we support:
                                         "NO2"
                                         "HCHO"
             ctm_models_coordinate [dict]: the ctm coordinates
             YYYYMM [int]: the target month and year, e.g., 202005 (May 2020)
             trop [bool]: true for considering the tropospheric region only
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal
             num_job [int]: the number of jobs for parallel computation
        Output [tropomi]: the tropomi @dataclass
    '''

    # find L2 files first
    L2_files = sorted(glob.glob(product_dir + "/S5P_*" + "_L2__*__" + str(YYYYMM) + "*.nc"))
    # read the files in parallel
    if satellite_product_name.split('_')[-1] == 'NO2':
        outputs_sat = Parallel(n_jobs=num_job)(delayed(tropomi_reader_no2)(
            L2_files[k], trop, ctm_models_coordinate=ctm_models_coordinate, read_ak=read_ak) for k in range(len(L2_files)))
    elif satellite_product_name.split('_')[-1] == 'HCHO':
        outputs_sat = Parallel(n_jobs=num_job)(delayed(tropomi_reader_hcho)(
            L2_files[k], ctm_models_coordinate=ctm_models_coordinate, read_ak=read_ak) for k in range(len(L2_files)))
    return outputs_sat


def omi_reader(product_dir: str, satellite_product_name: str, ctm_models_coordinate: dict, YYYYMM: str, trop: bool, read_ak=True, num_job=1):
    '''
        reading omi data
             product_dir [str]: the folder containing the tropomi data
             satellite_product_name [str]: so far we support:
                                         "NO2"
                                         "HCHO"
                                         "O3"
             ctm_models_coordinate [dict]: the ctm coordinates
             YYYYMM [int]: the target month and year, e.g., 202005 (May 2020)
             trop [bool]: true for considering the tropospheric region only
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal
             num_job [int]: the number of jobs for parallel computation
        Output [tropomi]: the tropomi @dataclass
    '''

    # find L2 files first
    if satellite_product_name.split('_')[-1] != 'O3':
        print(product_dir + "/*" + YYYYMM[0:4] + 'm' + YYYYMM[4::] + "*.nc")
        L2_files = sorted(glob.glob(product_dir + "/*" +
                                    YYYYMM[0:4] + 'm' + YYYYMM[4::] + "*.nc"))
    else:
        print(product_dir + "/*" + YYYYMM[0:4] + 'm' + YYYYMM[4::] + "*.he5")
        L2_files = sorted(glob.glob(product_dir + "/*" +
                                    YYYYMM[0:4] + 'm' + YYYYMM[4::] + "*.he5"))
    # read the files in parallel
    if satellite_product_name.split('_')[-1] == 'NO2':
        outputs_sat = Parallel(n_jobs=num_job)(delayed(omi_reader_no2)(
            L2_files[k], trop, ctm_models_coordinate=ctm_models_coordinate, read_ak=read_ak) for k in range(len(L2_files)))
    elif satellite_product_name.split('_')[-1] == 'HCHO':
        outputs_sat = Parallel(n_jobs=num_job)(delayed(omi_reader_hcho)(
            L2_files[k], ctm_models_coordinate=ctm_models_coordinate, read_ak=read_ak) for k in range(len(L2_files)))
    return outputs_sat


class readers(object):

    def __init__(self) -> None:
        # set the desired location based on control_free.yml
        with open('control_free.yml', 'r') as stream:
            try:
                ctrl_opts = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise Exception(exc)

        lat1 = ctrl_opts['latll']
        lat2 = ctrl_opts['latur']
        lon1 = ctrl_opts['lonll']
        lon2 = ctrl_opts['lonur']
        gridsize = ctrl_opts['gridsize']
        lon_grid = np.arange(lon1, lon2+gridsize, gridsize)
        lat_grid = np.arange(lat1, lat2+gridsize, gridsize)
        self.lons_grid, self.lats_grid = np.meshgrid(
            lon_grid.astype('float16'), lat_grid.astype('float16'))

    def add_satellite_data(self, product_name: str, product_dir: Path):
        '''
            add L2 data
            Input:
                product_name [str]: a string specifying the type of data to read:
                                   TROPOMI_NO2
                                   TROPOMI_HCHO
                                   OMI_NO2
                                   OMI_HCHO

                product_dir  [Path]: a path object describing the path of L2 files
        '''
        self.satellite_product_dir = product_dir
        self.satellite_product_name = product_name

    def add_ctm_data(self, product_name: int, product_dir: Path):
        '''
            add CTM data
            Input:
                product_name [str]: an string specifying the type of data to read:
                                "GMI"
                                "ECCOH"
                product_dir  [Path]: a path object describing the path of CTM files
        '''

        self.ctm_product_dir = product_dir
        self.ctm_product = product_name

    def read_satellite_data(self, YYYYMM: str, read_ak=True, trop=False, num_job=1):
        '''
            read L2 satellite data
            Input:
             YYYYMM [str]: the target month and year, e.g., 202005 (May 2020)
             read_ak [bool]: true for reading averaging kernels. this must be true for amf_recal
             trop[bool]: true for only including the tropospheric region (relevant for NO2 only)
             num_job [int]: the number of jobs for parallel computation
        '''
        satellite = self.satellite_product_name.split('_')[0]
        ctm_models_coordinate = {}
        # taking lats and lons from the free yaml
        ctm_models_coordinate["Latitude"] = self.lats_grid
        ctm_models_coordinate["Longitude"] = self.lons_grid
        if satellite == 'TROPOMI':
            self.sat_data = tropomi_reader(self.satellite_product_dir.as_posix(),
                                           self.satellite_product_name, ctm_models_coordinate,
                                           YYYYMM,  trop, read_ak=read_ak, num_job=num_job)
        elif satellite == 'OMI':
            self.sat_data = omi_reader(self.satellite_product_dir.as_posix(),
                                       self.satellite_product_name, ctm_models_coordinate,
                                       YYYYMM,  trop, read_ak=read_ak, num_job=num_job)
        else:
            raise Exception("the satellite is not supported, come tomorrow!")

    def read_ctm_data(self, YYYYMM: str, num_job=1):
        '''
            read ctm data
            Input:
             YYYYMM [str]: the target month and year, e.g., 202005 (May 2020)
             num_job [int]: the number of jobs for parallel computation
        '''
        YYYYMM2 = list(YYYYMM)
        if float(YYYYMM[0:4])>2020:
           YYYYMM2[0:4]="2019"
        YYYYMM2 = ''.join(YYYYMM2)
        self.ctm_data = GMI_reader(
            self.ctm_product_dir.as_posix(), str(YYYYMM2), num_job=num_job)


# testing
if __name__ == "__main__":
    sat_path = []
    sat_path.append(
        Path('/discover/nobackup/asouri/PROJECTS/PO3_ACMAP/omi_no2_PO3'))
    sat_path.append(
        Path('/discover/nobackup/asouri/PROJECTS/PO3_ACMAP/omi_hcho_PO3'))
    reader_obj = readers()
    reader_obj.add_ctm_data('GMI', Path(
        '/discover/nobackup/asouri/GITS/OI-SAT-GMI/oisatgmi/download_bucket/gmi/'))
    #reader_obj.read_ctm_data('200506', num_job=12)
    # NO2
    reader_obj.add_satellite_data(
        'OMI_NO2', sat_path[0])
    reader_obj.read_satellite_data(
        '200506', read_ak=False, trop=True, num_job=12)

    latitude = reader_obj.sat_data[0].latitude_center
    longitude = reader_obj.sat_data[0].longitude_center

    output = np.zeros((np.shape(latitude)[0], np.shape(
        latitude)[1], len(reader_obj.sat_data)))
    output2 = np.zeros_like(output)
    counter = -1
    for trop in reader_obj.sat_data:
        counter = counter + 1
        if trop is None:
            continue
        output[:, :, counter] = trop.vcd

    #output[output <= 0.0] = np.nan
    moutput = {}
    moutput["vcds"] = output
    moutput["lat"] = latitude
    moutput["lon"] = longitude
    savemat("vcds_omi.mat", moutput)
