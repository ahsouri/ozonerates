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


def minds_reader_core(filename: str):

    print("Currently reading: " + filename.split('/')[-1])
    # read the data
    ctmtype = "MINDS"
    # read coordinates
    lon = _read_nc(filename, 'lon')
    lat = _read_nc(filename, 'lat')
    lons_grid, lats_grid = np.meshgrid(lon, lat)
    latitude = lats_grid
    longitude = lons_grid
    # read time
    time_min_delta = _read_nc(filename, 'time')
    time_attr = _get_nc_attr(filename, 'time')
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
    # read pressure, temperature, PBL and hcho and no2 conc
    PBL = _read_nc(filename, 'PBLH')
    ZL = np.flip(_read_nc(filename, 'ZL'), axis=1)
    # read hcho and no2 profiles
    HCHO = np.flip(_read_nc(
        filename, 'CH2O'), axis=1)*1e9
    NO2 = np.flip(_read_nc(
        filename, 'NO2'), axis=1)*1e9
    surface_press = _read_nc(filename, 'PS').astype('float32')/100.0
    a0 = np.array([0.000000e00, 4.804826e-02, 6.593752e00, 1.313480e01, 1.961311e01, 2.609201e01,
                   3.257081e01, 3.898201e01, 4.533901e01, 5.169611e01, 5.805321e01, 6.436264e01,
                   7.062198e01, 7.883422e01, 8.909992e01, 9.936521e01, 1.091817e02, 1.189586e02,
                   1.286959e02, 1.429100e02, 1.562600e02, 1.696090e02, 1.816190e02, 1.930970e02,
                   2.032590e02, 2.121500e02, 2.187760e02, 2.238980e02, 2.243630e02, 2.168650e02,
                   2.011920e02, 1.769300e02, 1.503930e02, 1.278370e02, 1.086630e02, 9.236572e01,
                   7.851231e01, 6.660341e01, 5.638791e01, 4.764391e01, 4.017541e01, 3.381001e01,
                   2.836781e01, 2.373041e01, 1.979160e01, 1.645710e01, 1.364340e01, 1.127690e01,
                   9.292942e00, 7.619842e00, 6.216801e00, 5.046801e00, 4.076571e00, 3.276431e00,
                   2.620211e00, 2.084970e00, 1.650790e00, 1.300510e00, 1.019440e00, 7.951341e-01,
                   6.167791e-01, 4.758061e-01, 3.650411e-01, 2.785261e-01, 2.113490e-01, 1.594950e-01,
                   1.197030e-01, 8.934502e-02, 6.600001e-02, 4.758501e-02, 3.270000e-02, 2.000000e-02,
                   1.000000e-02])
    b0 = np.array([1.000000e00, 9.849520e-01, 9.634060e-01, 9.418650e-01, 9.203870e-01, 8.989080e-01,
                   8.774290e-01, 8.560180e-01, 8.346609e-01, 8.133039e-01, 7.919469e-01, 7.706375e-01,
                   7.493782e-01, 7.211660e-01, 6.858999e-01, 6.506349e-01, 6.158184e-01, 5.810415e-01,
                   5.463042e-01, 4.945902e-01, 4.437402e-01, 3.928911e-01, 3.433811e-01, 2.944031e-01,
                   2.467411e-01, 2.003501e-01, 1.562241e-01, 1.136021e-01, 6.372006e-02, 2.801004e-02,
                   6.960025e-03, 8.175413e-09, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00,
                   0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00,
                   0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00,
                   0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00,
                   0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00,
                   0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00,
                   0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00, 0.000000e00,
                   0.000000e00])
    p_mid = np.zeros_like(HCHO)
    for z in range(0, np.size(a0)-1):
        p_mid[:, z, :, :] = 0.5 * \
            ((a0[z] + b0[z]*surface_press) + (a0[z+1] + b0[z+1]*surface_press))
    p_edge = np.zeros(
        (np.shape(HCHO)[0], 73, np.shape(HCHO)[2], np.shape(HCHO)[3]))
    for z in range(0, np.size(a0)):
        p_edge[:, z, :, :] = (a0[z] + b0[z]*surface_press)
    # find PBLH in pressure
    PBLH = np.zeros_like(PBL)
    for i in range(0, np.shape(PBL)[0]):
        for j in range(0, np.shape(PBL)[1]):
            for k in range(0, np.shape(PBL)[2]):
                cost = abs(ZL[i, :, j, k].squeeze() - PBL[i, j, k])
                index_pbl = np.argmin(cost)
                PBLH[i, j, k] = p_mid[i, index_pbl, j, k]

    p_mid = p_mid[:, :25, :, :]
    p_edge = p_edge[:, :25, :, :]
    NO2 = NO2[:, :25, :, :]
    HCHO = HCHO[:, :25, :, :]
    ZL = ZL[:, :25, :, :]
    # shape up the ctm class
    mind_data = ctm_model(latitude, longitude, time,
                          NO2.astype('float16'), HCHO.astype(
                              'float16'), p_mid.astype('float16'),
                          p_edge.astype('float16'), ZL.astype('float16'), PBLH, ctmtype)
    return mind_data


def minds_reader(product_dir: str, YYYYMM: str,) -> list:
    '''
       MINDS reader
       Inputs:
             product_dir [str]: the folder containing the GMI data
             YYYYMM [str]: the target month and year, e.g., 202005 (May 2020)
       Output:
             minds_fields [ctm_model]: a dataclass format
    '''
    # read meteorological and chemical fields
    tavg3_3d_gas_files = sorted(
        glob.glob(product_dir + "/*tavg3_3d_gmi_Nv." + str(YYYYMM[0:4]) + "*.nc4"))
    # define gas profiles for saving
    outputs = []
    for file in tavg3_3d_gas_files:
        outputs.append(minds_reader_core(file))
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
    print('Colocating Aircraft and MINDS...')
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
        # for t1 in range(0, 2500):
        # find the closest day
        closest_index = np.argmin(np.abs(time_aircraft[t1] - time_ctm))
        # find the closest hour (this only works for 3-hourly frequency)
        closest_index_day = int(np.floor(closest_index/8.0))
        closest_index_hour = int(closest_index % 8)
        print("The closest MINDS file used for the aircraft at " + str(airdata.time[t1]) +
              " is at " + str(time_ctm_datetype[closest_index_day][closest_index_hour]))
        ctm_mid_pressure = ctmdata[closest_index_day].pressure_mid[closest_index_hour, :, :, :].squeeze(
        )
        ctm_pressure_edge = ctmdata[closest_index_day].pressure_edge[closest_index_hour, :, :, :].squeeze(
        )
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
        ctm_pressure_edge_new = ctm_pressure_edge[index_z, index_i, index_j].squeeze(
        )
        ctm_no2_profile_new = ctm_no2_profile[index_z,
                                              index_i, index_j].squeeze()
        ctm_hcho_profile_new = ctm_hcho_profile[index_z, index_i, index_j].squeeze(
        )
        ctm_ZL_new = ctm_ZL[index_z, index_i, index_j].squeeze()
        ctm_PBLH_new = ctm_PBLH[index_i, index_j].squeeze()

        # in case we have more than one grid box cloes to the aircraft point
        if np.size(ctm_mid_pressure_new) > 1:
            ctm_mid_pressure_new = np.mean(ctm_mid_pressure_new)
            ctm_pressure_edge_new = np.mean(ctm_pressure_edge_new)
            ctm_no2_profile_new = np.mean(ctm_no2_profile_new)
            ctm_hcho_profile_new = np.mean(ctm_hcho_profile_new)
            ctm_ZL_new = np.mean(ctm_ZL_new)

        if np.size(ctm_PBLH_new) > 1:
            ctm_PBLH_new = np.mean(ctm_PBLH_new)

        ctm_mapped_pressure.append(ctm_mid_pressure_new)
        ctm_mapped_pressure_edge.append(ctm_pressure_edge_new)
        ctm_mapped_pblh.append(ctm_PBLH_new)
        ctm_mapped_no2.append(ctm_no2_profile_new)
        ctm_mapped_hcho.append(ctm_hcho_profile_new)
        ctm_mapped_ZL.append(ctm_ZL_new)

    # converting the lists to numpy array
    ctm_mapped_pressure = np.array(ctm_mapped_pressure)
    ctm_mapped_pressure_edge = np.array(ctm_mapped_pressure_edge)
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

        # map aircraft and ctm  vertical grid into a regular vertical grid
        regular_grid_edge = np.flip(np.arange(450, 1015+20, 20))
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
            int(year_spiral), int(month_spiral), int(day_spiral), 12, 15, 0)
        time_leo_lst_approximate_up = datetime.datetime(
            int(year_spiral), int(month_spiral), int(day_spiral), 15, 10, 0)
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
            mask_PBL = spiral_data["pressure_mid"][spiral,
                                                   :] >= spiral_data["PBLH"][spiral]
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
    # TEXAS
    aircraft_path = './discoveraq-mrg15-p3b_merge_20130904_R3_thru20130929.ict'
    aircraft_data1 = aircraft_reader(
        aircraft_path, 2013, 'NO2_MixingRatio', 'CH2O_DFGAS')
    minds_data = minds_reader('./minds/', '201309')
    spiral_data = colocate(minds_data, aircraft_data1)
    to_mat(spiral_data, 'output_tx.mat')

    # COLORADO
    aircraft_path = './discoveraq-mrg10-p3b_merge_20140717_R2_thru20140810.ict'
    aircraft_data1 = aircraft_reader(
        aircraft_path, 2014, 'NO2_MixingRatio', 'CH2O_DFGAS')
    minds_data = minds_reader('./minds/', '201407')
    spiral_data = colocate(minds_data, aircraft_data1)
    to_mat(spiral_data, 'output_co.mat')

    # MD/DC
    aircraft_path = './discoveraq-mrg15-p3b_merge_20110701_R4_thru20110729.ict'
    aircraft_data1 = aircraft_reader(
        aircraft_path, 2011, 'NO2_NCAR', 'CH2O_DFGAS')
    minds_data = minds_reader('./minds/', '201107')
    spiral_data = colocate(minds_data, aircraft_data1)
    to_mat(spiral_data, 'output_md.mat')

    # CALIFORNIA
    aircraft_path = './discoveraq-mrg10-p3b_merge_20130116_R4_thru20130206.ict'
    aircraft_data1 = aircraft_reader(
        aircraft_path, 2013, 'NO2_MixingRatio', 'CH2O_DFGAS')
    minds_data = minds_reader('./minds/', '201301')
    spiral_data = colocate(minds_data, aircraft_data1)
    to_mat(spiral_data, 'output_ca.mat')

    # KORUS
    aircraft_path = './korusaq-mrg10-dc8_merge_20160426_R6_thru20160618.ict'
    aircraft_data1 = aircraft_reader(
        aircraft_path, 2016, 'NO2_MixingRatio', 'CH2O_CAMS')
    minds_data = minds_reader('./minds/', '201605')
    spiral_data = colocate(minds_data, aircraft_data1)
    to_mat(spiral_data, 'output_kor.mat')
