import numpy as np
import datetime
from dataclasses import dataclass
from netCDF4 import Dataset
import warnings
import glob
from scipy import interpolate
from scipy.io import savemat
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class aircraft_data:
    lat: np.ndarray
    lon: np.ndarray
    time: datetime.datetime
    HCHO_ppbv: np.ndarray
    NO2_ppbv: np.ndarray
    altp: np.ndarray
    profile_num: np.ndarray


@dataclass
class ctm_model:
    latitude: np.ndarray
    longitude: np.ndarray
    time: list
    gas_profile_no2: np.ndarray
    gas_profile_hcho: np.ndarray
    pressure_mid: np.ndarray
    pressure_edge: np.ndarray
    ZL: np.ndarray
    PBLH: np.ndarray
    ctmtype: str


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


def aircraft_reader(filename: str, year: int, NO2_string: str, HCHO_string: str):
    # read the header
    with open(filename) as f:
        header = f.readline().split(',')
    # read the data
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    doy = data[:, header.index(' JDAY')]
    utc_time = data[:, header.index('  UTC')]
    lat = data[:, header.index(' LATITUDE')]
    lon = data[:, header.index(' LONGITUDE')]
    lon[lon > 180.0] = lon[lon > 180.0] - 360.0
    HCHO_ppbv = data[:, header.index(' ' + HCHO_string)]/1000.0
    HCHO_ppbv[HCHO_ppbv <= 0] = np.nan
    NO2_ppbv = data[:, header.index(' ' + NO2_string)]/1000.0
    NO2_ppbv[NO2_ppbv <= 0] = np.nan
    altp = data[:, header.index(' PRESSURE')]
    altp[altp <= 0] = np.nan
    profile_num = data[:, header.index(' ProfileNumber')]
    # convert doy and utc to datetime
    date = []
    for i in range(0, np.size(doy)):
        date.append(datetime.datetime.strptime(str(int(year)) + "-" + str(int(doy[i])), "%Y-%j") +
                    datetime.timedelta(seconds=int(utc_time[i])))
    # return a structure
    return aircraft_data(lat, lon, date, HCHO_ppbv, NO2_ppbv, altp, profile_num)

def aircraft_aeromma_reader(filename: str, NO2_string: str, HCHO_string: str):
    # read the header
    data = pd.read_csv(filename)
    # read the data
    lat = np.array(data["G_LAT"])
    lon = np.array(data["G_LONG"])
    lon[lon > 180.0] = lon[lon > 180.0] - 360.0
    HCHO_ppbv = np.array(data[HCHO_string])/1000.0
    HCHO_ppbv[HCHO_ppbv <= 0] = np.nan
    NO2_ppbv = np.array(data[NO2_string])/1000.0
    NO2_ppbv[NO2_ppbv <= 0] = np.nan
    altp = np.array(data["P"])
    altp[altp <= 0] = np.nan
    profile_num = np.array(data["profile_number"])
    # convert doy and utc to datetime
    date = []
    date = pd.to_datetime(data["Time"]).to_list()
    # return a structure
    return aircraft_data(lat, lon, date, HCHO_ppbv, NO2_ppbv, altp, profile_num)

def cmaq_reader_core(cmaq_target_file,met_file_3d_file,met_file_2d_file,grd_file_2d_file):
        
        print("Currently reading: " + cmaq_target_file.split('/')[-1])
        # reading time and coordinates
        lat = _read_nc(grd_file_2d_file, 'LAT')
        lon = _read_nc(grd_file_2d_file, 'LON')
        time_var = _read_nc(cmaq_target_file, 'TFLAG')
        # populating cmaq time
        time = []
        for t in range(0, np.shape(time_var)[0]):
            cmaq_date = datetime.datetime.strptime(
                str(time_var[t, 0, 0]), '%Y%j').date()
            time.append(datetime.datetime(int(cmaq_date.strftime('%Y')), int(cmaq_date.strftime('%m')),
                                      int(cmaq_date.strftime('%d')), int(time_var[t, 0, 1]/10000.0), 0, 0) +
                    datetime.timedelta(minutes=0))

        prs = _read_nc(met_file_3d_file, 'PRES').astype('float32')/100.0  # hPa
        PBLH = _read_nc(met_file_2d_file, 'PBL').astype('float32')
        ZL = _read_nc(met_file_3d_file, 'ZH').astype('float32')
        # read gas in ppbv
        gas_hcho = _read_nc(cmaq_target_file, 'FORM')*1000.0  # ppb
        gas_hcho = gas_hcho.astype('float32')
        gas_no2 = _read_nc(cmaq_target_file, 'NO2')*1000.0  # ppb
        gas_no2 = gas_no2.astype('float32')        
        # populate cmaq_data format
        cmaq_data = ctm_model(lat, lon, time, gas_no2, gas_hcho, prs, [], ZL, PBLH, 'CMAQ')
        return cmaq_data

def cmaq_reader(dir_mcip: str, dir_cmaq: str, YYYYMMDD: str) -> list:
    '''
       MINDS reader
       Inputs:
             product_dir [str]: the folder containing the GMI data
             YYYYMM [str]: the target month and year, e.g., 202005 (May 2020)
       Output:
             minds_fields [ctm_model]: a dataclass format
    '''
    # read meteorological and chemical fields
    cmaq_target_files = sorted(glob.glob(dir_cmaq + "/CCTM_CONC_*" + YYYYMMDD +  "*.nc"))
    grd_files_2d = sorted(
            glob.glob(dir_mcip + "/GRIDCRO2D_*" + \
        YYYYMMDD +  "*"))
    met_files_2d = sorted(
            glob.glob(dir_mcip + "/METCRO2D_*" + YYYYMMDD  + "*"))
    met_files_3d = sorted(
            glob.glob(dir_mcip + "/METCRO3D_*" + YYYYMMDD  + "*"))
    if len(cmaq_target_files) != len(met_files_3d):
            raise Exception(
                "the data are not consistent")

    # define gas profiles for saving
    outputs = []
    for k in range(len(met_files_3d)):
        outputs.append(cmaq_reader_core(cmaq_target_files[k],met_files_3d[k],met_files_2d[k],grd_files_2d[k]))

    return outputs

def colocate(ctmdata, airdata):
    '''
       Colocate the model and the aircraft based on the nearest neighbour
       Inputs:
             ctmdata: structure for the model
             airdata: structure for the aircraft dataset
       Output:
             a dict containing colocated model and aircraft
    '''
    print('Colocating Aircraft and CMAQ...')
    # list the time in ctm_data
    time_ctm = []
    time_ctm_datetype = []
    for ctm_granule in ctmdata:
        time_temp = ctm_granule.time
        for n in range(len(time_temp)):
            time_temp2 = time_temp[n].year*10000 + time_temp[n].month*100 +\
                time_temp[n].day + time_temp[n].hour/24.0 + \
                time_temp[n].minute/60.0/24.0 + time_temp[n].second/3600.0/24.0
            time_ctm.append(time_temp2)
        time_ctm_datetype.append(ctm_granule.time)

    time_ctm = np.array(time_ctm)

    time_aircraft = []
    for t in airdata.time:
        time_aircraft.append(t.year*10000 + t.month*100 +
                             t.day + t.hour/24.0 + t.minute /
                             60.0/24.0 + t.second/3600.0/24.0)

    time_aircraft = np.array(time_aircraft)

    # translating ctm data into aircraft location/time
    ctm_mapped_pressure = []
    ctm_mapped_pressure_edge = []
    ctm_mapped_no2 = []
    ctm_mapped_hcho = []
    ctm_mapped_pblh = []
    ctm_mapped_ZL = []
    for t1 in range(0, np.size(airdata.time)):
        #for t1 in range(0, 2500):
        if np.isnan(airdata.lat[t1]):
           ctm_mapped_pressure.append(np.nan)
           #ctm_mapped_pressure_edge.append(ctm_pressure_edge_new)
           ctm_mapped_pblh.append(np.nan)
           ctm_mapped_no2.append(np.nan)
           ctm_mapped_hcho.append(np.nan)
           ctm_mapped_ZL.append(np.nan)
           continue
        # find the closest day
        closest_index = np.argmin(np.abs(time_aircraft[t1] - time_ctm))
        # find the closest hour (this only works for 3-hourly frequency)
        closest_index_day = int(np.floor(closest_index/25.0))
        closest_index_hour = int(closest_index % 25)
        print("The closest MINDS file used for the aircraft at " + str(airdata.time[t1]) +
              " is at " + str(time_ctm_datetype[closest_index_day][closest_index_hour]))
        ctm_mid_pressure = ctmdata[closest_index_day].pressure_mid[closest_index_hour, :, :, :].squeeze(
        )
        #ctm_pressure_edge = ctmdata[closest_index_day].pressure_edge[closest_index_hour, :, :, :].squeeze()
        ctm_no2_profile = ctmdata[closest_index_day].gas_profile_no2[closest_index_hour, :, :].squeeze(
        )
        ctm_hcho_profile = ctmdata[closest_index_day].gas_profile_hcho[closest_index_hour, :, :].squeeze(
        )
        ctm_PBLH = ctmdata[closest_index_day].PBLH[closest_index_hour, :, :].squeeze(
        )
        ctm_ZL = ctmdata[closest_index_day].ZL[closest_index_hour, :, :].squeeze(
        )

        # pinpoint the closest location based on NN
        cost = np.sqrt((airdata.lon[t1]-ctmdata[0].longitude)
                       ** 2 + (airdata.lat[t1]-ctmdata[0].latitude)**2)
        index_i, index_j = np.where(cost == min(cost.flatten()))
        # pinpoint the closet altitude pressure based on NN
        cost = np.abs(
            airdata.altp[t1]-ctm_mid_pressure[:, index_i, index_j].squeeze())
        index_z = np.argmin(cost)
        # picking the right grid
        ctm_mid_pressure_new = ctm_mid_pressure[index_z, index_i, index_j].squeeze(
        )
        #ctm_pressure_edge_new = ctm_pressure_edge[index_z, index_i, index_j].squeeze()
        ctm_no2_profile_new = ctm_no2_profile[index_z,
                                              index_i, index_j].squeeze()
        ctm_hcho_profile_new = ctm_hcho_profile[index_z, index_i, index_j].squeeze(
        )
        ctm_ZL_new = ctm_ZL[index_z, index_i, index_j].squeeze()
        ctm_PBLH_new = ctm_PBLH[index_i, index_j].squeeze()

        # in case we have more than one grid box cloes to the aircraft point
        if np.size(ctm_mid_pressure_new) > 1:
            ctm_mid_pressure_new = np.mean(ctm_mid_pressure_new)
            #ctm_pressure_edge_new = np.mean(ctm_pressure_edge_new)
            ctm_no2_profile_new = np.mean(ctm_no2_profile_new)
            ctm_hcho_profile_new = np.mean(ctm_hcho_profile_new)
            ctm_ZL_new = np.mean(ctm_ZL_new)

        if np.size(ctm_PBLH_new) > 1:
            ctm_PBLH_new = np.mean(ctm_PBLH_new)

        ctm_mapped_pressure.append(ctm_mid_pressure_new)
        #ctm_mapped_pressure_edge.append(ctm_pressure_edge_new)
        ctm_mapped_pblh.append(ctm_PBLH_new)
        ctm_mapped_no2.append(ctm_no2_profile_new)
        ctm_mapped_hcho.append(ctm_hcho_profile_new)
        ctm_mapped_ZL.append(ctm_ZL_new)

    # converting the lists to numpy array
    ctm_mapped_pressure = np.array(ctm_mapped_pressure)
    #ctm_mapped_pressure_edge = np.array(ctm_mapped_pressure_edge)
    ctm_mapped_pblh = np.array(ctm_mapped_pblh)
    ctm_mapped_no2 = np.array(ctm_mapped_no2)
    ctm_mapped_hcho = np.array(ctm_mapped_hcho)
    ctm_mapped_ZL = np.array(ctm_mapped_ZL)

    # now colocating based on each spiral number
    unique = np.unique(airdata.profile_num)
    output = {}
    output["HCHO_profile_aircraft"] = []
    output["NO2_profile_aircraft"] = []
    output["HCHO_profile_model"] = []
    output["NO2_profile_model"] = []
    output["pressure_mid"] = []
    output["dp"] = []
    output["PBLH"] = []
    output["Spiral_num"] = []
    output["time"] = []
    output["lon"] = []
    output["ZL"] = []

    # looping over unique spirals
    for u in range(0, np.size(unique)):
        if unique[u] == 0:  # skip bad spirals
            continue

        # take aircraft data for this specific spiral
        aircraft_pres_unique = airdata.altp[np.where(
            airdata.profile_num == unique[u])]
        aircraft_HCHO_unique = airdata.HCHO_ppbv[np.where(
            airdata.profile_num == unique[u])]
        aircraft_NO2_unique = airdata.NO2_ppbv[np.where(
            airdata.profile_num == unique[u])]
        aircraft_time_unique = time_aircraft[np.where(
            airdata.profile_num == unique[u])]
        aircraft_lon_unique = airdata.lon[np.where(
            airdata.profile_num == unique[u])]

        # take model fields for this specific spiral
        ctm_press_chosen = ctm_mapped_pressure[np.where(
            airdata.profile_num == unique[u])]
        ctm_pblh_chosen = np.nanmean(
            ctm_mapped_pblh[np.where(airdata.profile_num == unique[u])]).squeeze()
        ctm_hcho_chosen = ctm_mapped_hcho[np.where(
            airdata.profile_num == unique[u])]
        ctm_no2_chosen = ctm_mapped_no2[np.where(
            airdata.profile_num == unique[u])]
        ctm_ZL_chosen = ctm_mapped_ZL[np.where(
            airdata.profile_num == unique[u])]
        # sometimes the aircraft pressure doesn't steadily increase/decrease, sort them
        index_sort = np.argsort(aircraft_pres_unique)
        aircraft_pres_unique = aircraft_pres_unique[index_sort]
        aircraft_HCHO_unique = aircraft_HCHO_unique[index_sort]
        aircraft_NO2_unique = aircraft_NO2_unique[index_sort]
        aircraft_time_unique = aircraft_time_unique[index_sort]
        ctm_no2_chosen = ctm_no2_chosen[index_sort]
        ctm_hcho_chosen = ctm_hcho_chosen[index_sort]
        ctm_pres_chosen = ctm_press_chosen[index_sort]
        ctm_ZL_chosen = ctm_ZL_chosen[index_sort]

        # throw out bad data
        ctm_no2_chosen[np.isnan(aircraft_NO2_unique)] = np.nan
        ctm_hcho_chosen[np.isnan(aircraft_HCHO_unique)] = np.nan
        if np.size(ctm_no2_chosen) == 0:
           continue
        # map aircraft and ctm  vertical grid into a regular vertical grid
        regular_grid_edge = np.flip(np.arange(450, 1015+20, 10))
        regular_grid_mid = np.zeros(np.size(regular_grid_edge)-1)
        dp = np.zeros_like(regular_grid_mid)
        for z in range(0, np.size(regular_grid_edge)-1):
            regular_grid_mid[z] = 0.5 * \
                (regular_grid_edge[z]+regular_grid_edge[z+1])
            dp[z] = regular_grid_edge[z] - regular_grid_edge[z+1]

        f = interpolate.interp1d(
            np.log(aircraft_pres_unique),
            aircraft_HCHO_unique, bounds_error=False, fill_value=np.nan)

        interpolated_HCHO = f(np.log(regular_grid_mid)).squeeze()

        f = interpolate.interp1d(
            np.log(aircraft_pres_unique),
            aircraft_NO2_unique, bounds_error=False, fill_value=np.nan)

        interpolated_NO2 = f(np.log(regular_grid_mid)).squeeze()

        f = interpolate.interp1d(
            np.log(aircraft_pres_unique),
            aircraft_time_unique, bounds_error=False, fill_value=np.nan)

        interpolated_time = f(np.log(regular_grid_mid)).squeeze()

        f = interpolate.interp1d(
            np.log(aircraft_pres_unique),
            aircraft_lon_unique, bounds_error=False, fill_value=np.nan)

        interpolated_lon = f(np.log(regular_grid_mid)).squeeze()

        f = interpolate.interp1d(
            np.log(ctm_pres_chosen),
            ctm_no2_chosen, bounds_error=False, fill_value=np.nan)

        interpolated_NO2_model = f(np.log(regular_grid_mid)).squeeze()

        f = interpolate.interp1d(
            np.log(ctm_pres_chosen),
            ctm_hcho_chosen, bounds_error=False, fill_value=np.nan)

        interpolated_HCHO_model = f(np.log(regular_grid_mid)).squeeze()

        f = interpolate.interp1d(
            np.log(ctm_pres_chosen),
            ctm_ZL_chosen, bounds_error=False, fill_value=np.nan)

        interpolated_ZL_model = f(np.log(regular_grid_mid)).squeeze()

        output["HCHO_profile_aircraft"].append(interpolated_HCHO)
        output["NO2_profile_aircraft"].append(interpolated_NO2)
        output["NO2_profile_model"].append(interpolated_NO2_model)
        output["HCHO_profile_model"].append(interpolated_HCHO_model)
        output["pressure_mid"].append(regular_grid_mid)
        output["dp"].append(dp)
        output["Spiral_num"].append(unique[u])
        output["PBLH"].append(ctm_pblh_chosen)
        output["time"].append(interpolated_time)
        output["lon"].append(np.nanmean(interpolated_lon))
        output["ZL"].append(interpolated_ZL_model)

    output["HCHO_profile_aircraft"] = np.array(output["HCHO_profile_aircraft"])
    output["NO2_profile_aircraft"] = np.array(output["NO2_profile_aircraft"])
    output["pressure_mid"] = np.array(output["pressure_mid"])
    output["dp"] = np.array(output["dp"])
    output["Spiral_num"] = np.array(output["Spiral_num"])
    output["NO2_profile_model"] = np.array(output["NO2_profile_model"])
    output["HCHO_profile_model"] = np.array(output["HCHO_profile_model"])
    output["time"] = np.array(output["time"])
    output["lon"] = np.array(output["lon"])
    output["PBLH"] = np.array(output["PBLH"])
    output["ZL"] = np.array(output["ZL"])
    return output

def colocate_nonspiral(ctmdata, airdata):
    '''
       Colocate the model and the aircraft based on the nearest neighbour
       Inputs:
             ctmdata: structure for the model
             airdata: structure for the aircraft dataset
       Output:
             a dict containing colocated model and aircraft
    '''
    print('Colocating Aircraft and CMAQ...')
    # list the time in ctm_data
    time_ctm = []
    time_ctm_datetype = []
    for ctm_granule in ctmdata:
        time_temp = ctm_granule.time
        for n in range(len(time_temp)):
            time_temp2 = time_temp[n].year*10000 + time_temp[n].month*100 +\
                time_temp[n].day + time_temp[n].hour/24.0 + \
                time_temp[n].minute/60.0/24.0 + time_temp[n].second/3600.0/24.0
            time_ctm.append(time_temp2)
        time_ctm_datetype.append(ctm_granule.time)

    time_ctm = np.array(time_ctm)

    time_aircraft = []
    for t in airdata.time:
        time_aircraft.append(t.year*10000 + t.month*100 +
                             t.day + t.hour/24.0 + t.minute /
                             60.0/24.0 + t.second/3600.0/24.0)

    time_aircraft = np.array(time_aircraft)
    # translating ctm data into aircraft location/time
    ctm_mapped_pressure = []
    ctm_mapped_pressure_edge = []
    ctm_mapped_no2 = []
    ctm_mapped_hcho = []
    ctm_mapped_pblh = []
    ctm_mapped_ZL = []
    for t1 in range(0, np.size(airdata.time)):
        if np.isnan(airdata.lat[t1]):
           ctm_mapped_pressure.append(-999.0)
           #ctm_mapped_pressure_edge.append(ctm_pressure_edge_new)
           ctm_mapped_pblh.append(-999.0)
           ctm_mapped_no2.append(-999.0)
           ctm_mapped_hcho.append(-999.0)
           ctm_mapped_ZL.append(-999.0)
           continue
        # for t1 in range(0, 2500):
        # find the closest day
        closest_index = np.argmin(np.abs(time_aircraft[t1] - time_ctm))
        # find the closest hour (this only works for 3-hourly frequency)
        closest_index_day = int(np.floor(closest_index/25.0))
        closest_index_hour = int(closest_index % 25)
        print("The closest CMAQ file used for the aircraft at " + str(airdata.time[t1]) +
              " is at " + str(time_ctm_datetype[closest_index_day][closest_index_hour]))
        ctm_mid_pressure = ctmdata[closest_index_day].pressure_mid[closest_index_hour, :, :, :].squeeze(
        )
        #ctm_pressure_edge = ctmdata[closest_index_day].pressure_edge[closest_index_hour, :, :, :].squeeze()
        ctm_no2_profile = ctmdata[closest_index_day].gas_profile_no2[closest_index_hour, :, :].squeeze(
        )
        ctm_hcho_profile = ctmdata[closest_index_day].gas_profile_hcho[closest_index_hour, :, :].squeeze(
        )
        ctm_PBLH = ctmdata[closest_index_day].PBLH[closest_index_hour, :, :].squeeze(
        )
        ctm_ZL = ctmdata[closest_index_day].ZL[closest_index_hour, :, :].squeeze(
        )

        # pinpoint the closest location based on NN
        cost = np.sqrt((airdata.lon[t1]-ctmdata[0].longitude)
                       ** 2 + (airdata.lat[t1]-ctmdata[0].latitude)**2)
        index_i, index_j = np.where(cost == min(cost.flatten()))
        # pinpoint the closet altitude pressure based on NN
        cost = np.abs(
            airdata.altp[t1]-ctm_mid_pressure[:, index_i, index_j].squeeze())
        index_z = np.argmin(cost)
        # picking the right grid
        ctm_mid_pressure_new = ctm_mid_pressure[index_z, index_i, index_j].squeeze(
        )
        #ctm_pressure_edge_new = ctm_pressure_edge[index_z, index_i, index_j].squeeze()
        ctm_no2_profile_new = ctm_no2_profile[index_z,
                                              index_i, index_j].squeeze()
        ctm_hcho_profile_new = ctm_hcho_profile[index_z, index_i, index_j].squeeze(
        )
        ctm_ZL_new = ctm_ZL[index_z, index_i, index_j].squeeze()
        ctm_PBLH_new = ctm_PBLH[index_i, index_j].squeeze()

        # in case we have more than one grid box cloes to the aircraft point
        if np.size(ctm_mid_pressure_new) > 1:
            ctm_mid_pressure_new = np.mean(ctm_mid_pressure_new)
            #ctm_pressure_edge_new = np.mean(ctm_pressure_edge_new)
            ctm_no2_profile_new = np.mean(ctm_no2_profile_new)
            ctm_hcho_profile_new = np.mean(ctm_hcho_profile_new)
            ctm_ZL_new = np.mean(ctm_ZL_new)

        if np.size(ctm_PBLH_new) > 1:
            ctm_PBLH_new = np.mean(ctm_PBLH_new)

        ctm_mapped_pressure.append(ctm_mid_pressure_new)
        #ctm_mapped_pressure_edge.append(ctm_pressure_edge_new)
        ctm_mapped_pblh.append(ctm_PBLH_new)
        ctm_mapped_no2.append(ctm_no2_profile_new)
        ctm_mapped_hcho.append(ctm_hcho_profile_new)
        ctm_mapped_ZL.append(ctm_ZL_new)

    # converting the lists to numpy array
    ctm_mapped_pressure = np.array(ctm_mapped_pressure)
    #ctm_mapped_pressure_edge = np.array(ctm_mapped_pressure_edge)
    ctm_mapped_pblh = np.array(ctm_mapped_pblh)
    ctm_mapped_no2 = np.array(ctm_mapped_no2)
    ctm_mapped_hcho = np.array(ctm_mapped_hcho)
    ctm_mapped_ZL = np.array(ctm_mapped_ZL)

    # now colocating based on each spiral number
    output = {}
    output["HCHO_profile_aircraft"] = airdata.HCHO_ppbv
    output["NO2_profile_aircraft"] = airdata.NO2_ppbv
    output["HCHO_profile_model"] = ctm_mapped_hcho
    output["NO2_profile_model"] = ctm_mapped_no2
    output["pressure_mid"] = ctm_mapped_pressure
    output["dp"] = []
    output["PBLH"] = ctm_mapped_pblh
    output["Spiral_num"] = []
    output["time"] = time_aircraft
    output["lon"] = airdata.lon
    output["lat"] = airdata.lat
    output["ZL"] = ctm_mapped_ZL
    savemat('./AEROMMA_test.mat', output)
    return None

def to_mat(spiral_data, filename):

    pressure = []
    NO2_air = []
    NO2_model = []
    HCHO_air = []
    HCHO_model = []
    no2_conversion_model = []
    no2_conversion_air = []
    hcho_conversion_air = []
    hcho_conversion_model = []
    pbl_mask = []
    height = []
    for spiral in range(0, np.shape(spiral_data["Spiral_num"])[0]):
        # define if the time is close to 1:45+-1.5 hours
        time_spiral = np.nanmean(spiral_data["time"][spiral, :]).squeeze()
        print(time_spiral)
        if np.isnan(time_spiral):
            continue
        year_spiral = np.floor(time_spiral/10000.0)
        month_spiral = np.floor((time_spiral-year_spiral*10000.0)/100.0)
        day_spiral = np.floor(time_spiral-year_spiral*10000.0-month_spiral*100)
        frac_spiral = time_spiral - \
            (year_spiral*10000.0+month_spiral*100+day_spiral)
        frac_spiral = frac_spiral*24*60  # minutues
        hour_spiral = np.floor(frac_spiral/60.0)
        min_spiral = frac_spiral - hour_spiral*60
        time_spiral_utc = datetime.datetime(int(year_spiral), int(
            month_spiral), int(day_spiral), int(hour_spiral), int(min_spiral), 0)
        # convert utc to local time (approximate)
        seconds_lon = (spiral_data["lon"][spiral]*3600)/15
        time_spiral_lst = time_spiral_utc + \
            datetime.timedelta(seconds=int(seconds_lon))
        time_leo_lst_approximate_lb = datetime.datetime(
            int(year_spiral), int(month_spiral), int(day_spiral), 12, 0, 0)
        time_leo_lst_approximate_up = datetime.datetime(
            int(year_spiral), int(month_spiral), int(day_spiral), 23, 59, 0)
        # if +- 1.5 hours from 1:45 LST
        if (time_spiral_lst >= time_leo_lst_approximate_lb) and (time_spiral_lst <= time_leo_lst_approximate_up):
            print(datetime.datetime.strptime(
                str(time_spiral_lst), "%Y-%m-%d %H:%M:%S"))
            pressure.append(spiral_data["pressure_mid"][spiral, :])
            NO2_air.append(spiral_data["NO2_profile_aircraft"][spiral, :])
            HCHO_air.append(spiral_data["HCHO_profile_aircraft"][spiral, :])
            # conversion calculation
            air_NO2 = spiral_data["NO2_profile_aircraft"][spiral, :]
            ctm_NO2 = spiral_data["NO2_profile_model"][spiral, :]
            air_HCHO = spiral_data["HCHO_profile_aircraft"][spiral, :]
            ctm_HCHO = spiral_data["HCHO_profile_model"][spiral, :]
            ctm_ZL = spiral_data["ZL"][spiral, :]
            ctm_NO2[np.isnan(air_NO2)] = np.nan
            ctm_HCHO[np.isnan(air_HCHO)] = np.nan
            air_HCHO[np.isnan(ctm_HCHO)] = np.nan
            air_NO2[np.isnan(ctm_NO2)] = np.nan
            NO2_model.append(ctm_NO2)
            HCHO_model.append(ctm_HCHO)
            dp = spiral_data["dp"][spiral, :]
            Mair = 28.97e-3
            g = 9.80665
            N_A = 6.02214076e23
            mask_PBL = spiral_data["ZL"][spiral,
                                                   :] <= spiral_data["PBLH"][spiral]
            mask_PBL = np.multiply(mask_PBL, 1.0).squeeze()
            mask_PBL[mask_PBL != 1.0] = np.nan
            no2_conversion_air.append(
                np.nanmean(1e9*mask_PBL*air_NO2)/np.nansum(air_NO2*dp/g/Mair*N_A*1e-4*100.0*1e-15))
            no2_conversion_model.append(np.nanmean(
                1e9*mask_PBL*ctm_NO2)/np.nansum(ctm_NO2*dp/g/Mair*N_A*1e-4*100.0*1e-15))
            hcho_conversion_air.append(np.nanmean(
                1e9*mask_PBL*air_HCHO)/np.nansum(air_HCHO*dp/g/Mair*N_A*1e-4*100.0*1e-15))
            hcho_conversion_model.append(np.nanmean(
                1e9*mask_PBL*ctm_HCHO)/np.nansum(ctm_HCHO*dp/g/Mair*N_A*1e-4*100.0*1e-15))
            pbl_mask.append(mask_PBL)
            height.append(ctm_ZL)

    pressure = np.array(pressure)
    HCHO_air = np.array(HCHO_air)
    HCHO_model = np.array(HCHO_model)
    NO2_air = np.array(NO2_air)
    NO2_model = np.array(NO2_model)
    no2_conversion_air = np.array(no2_conversion_air)
    no2_conversion_model = np.array(no2_conversion_model)
    hcho_conversion_air = np.array(hcho_conversion_air)
    hcho_conversion_model = np.array(hcho_conversion_model)
    pbl_mask = np.array(pbl_mask)
    height = np.array(height)

    output = {}
    output["pressure"] = pressure
    output["HCHO_air"] = HCHO_air
    output["HCHO_model"] = HCHO_model
    output["NO2_air"] = NO2_air
    output["NO2_model"] = NO2_model
    output["no2_conversion_air"] = no2_conversion_air
    output["no2_conversion_model"] = no2_conversion_model
    output["hcho_conversion_air"] = hcho_conversion_air
    output["hcho_conversion_model"] = hcho_conversion_model
    output["pbl_mask"] = pbl_mask
    output["height"] = height
    savemat(filename, output)

    # plotting for debugging
    pressure_mean = np.nanmean(np.array(pressure), axis=0)
    HCHO_air = np.nanmean(np.array(HCHO_air), axis=0)
    HCHO_air_std = np.nanstd(np.array(HCHO_air), axis=0)
    HCHO_model = np.nanmean(np.array(HCHO_model), axis=0)
    HCHO_model_std = np.nanstd(np.array(HCHO_model), axis=0)

    fig = plt.figure(figsize=(16, 8))
    sns.set()
    x = pressure_mean
    mean_1 = HCHO_air
    std_1 = HCHO_air_std

    mean_2 = HCHO_model
    std_2 = HCHO_model_std

    line_1, = plt.plot(x, mean_1, 'b-')
    fill_1 = plt.fill_between(
        x, mean_1 - std_1, mean_1 + std_1, color='b', alpha=0.2)
    line_2, = plt.plot(x, mean_2, 'r--')
    fill_2 = plt.fill_between(
        x, mean_2 - std_2, mean_2 + std_2, color='r', alpha=0.2)
    plt.margins(x=0)

    plt.legend([(line_1, fill_1), (line_2, fill_2)], [
               'Series 1', 'Series 2'], title='title')
    plt.legend(title='title')
    plt.tight_layout()
    fig.savefig('./test.png', format='png', dpi=300)
    plt.close()


if __name__ == "__main__":
    aeromma_files = sorted(glob.glob("./aeromma_data/A*.csv"))
    for file in aeromma_files:
        try:
           aircraft_data1 = aircraft_aeromma_reader(
               file, 'NO2_LIF', 'CH2O_ISAF')
           split_file = file.split('_')
           cmaq_data = cmaq_reader('./cmaq_test/','./cmaq_test/', split_file[2])
           spiral_data = colocate(cmaq_data, aircraft_data1)
           to_mat(spiral_data, 'output_aeromma_' + split_file[2] + '_PM.mat')
           spiral_data = []
           cmaq_data = []
           aircraft_data1 = []
        except:
           print("something bad happened")
