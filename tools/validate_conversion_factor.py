import numpy as np
import datetime
from dataclasses import dataclass
from netCDF4 import Dataset
import warnings
import glob
from scipy.spatial import Delaunay
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.interpolate import RBFInterpolator
from scipy import interpolate
from scipy.io import savemat

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
    delp: np.ndarray
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

def _interpolosis(interpol_func, Z: np.array, X: np.array, Y: np.array, interpolator_type: int, dists: np.array, threshold: float) -> np.array:
    # to make the interpolator() shorter
    if interpolator_type == 1:
        interpolator = LinearNDInterpolator(
            interpol_func, (Z).flatten(), fill_value=np.nan)
        ZZ = interpolator((X, Y))
        ZZ[dists > threshold*3.0] = np.nan
    elif interpolator_type == 2:
        interpolator = NearestNDInterpolator(interpol_func, (Z).flatten())
        ZZ = interpolator((X, Y))
        ZZ[dists > threshold*3.0] = np.nan
    elif interpolator_type == 3:
        interpolator = RBFInterpolator(
            interpol_func, (Z).flatten(), neighbors=5)
        XX = np.stack([X.ravel(), Y.ravel()], -1)
        ZZ = interpolator(XX)
        ZZ = ZZ.reshape(np.shape(X))
        ZZ[dists > threshold*3.0] = np.nan
    else:
        raise Exception(
            "other type of interpolation methods has not been implemented yet")
    return ZZ

def aircraft_reader(filename: str, year: int):
    # read the header
    with open(filename) as f:
        header = f.readline().split(',')
    # read the data
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    doy = data[:, header.index(' JDAY')]
    utc_time = data[:, header.index('  UTC')]
    lat = data[:, header.index(' LATITUDE')]
    lon = data[:, header.index(' LONGITUDE')]
    HCHO_ppbv = data[:, header.index(' CH2O_DFGAS')]/1000.0
    NO2_ppbv = data[:, header.index(' NO2_MixingRatio')]/1000.0
    altp = data[:, header.index(' PRESSURE')]
    profile_num = data[:, header.index(' ProfileNumber')]
    # conver doy and utc to datetime
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
    PBL = _read_nc(filename, 'PBLH')/100.0
    DELP = np.flip(_read_nc(filename, 'DELP')/100.0, axis=1)
    ZL = np.flip(_read_nc(filename, 'ZL'), axis=1)   
    # read hcho and no2 profiles
    HCHO = np.flip(_read_nc(
        filename, 'CH2O'), axis=1)
    NO2 = np.flip(_read_nc(
        filename, 'NO2'), axis=1)
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

    # find PBLH in pressure
    PBLH = np.zeros(PBL)
    for i in range(0,np.shape(PBL)[0]):
        for j in range(0,np.shape(PBL)[1]):
            for k in range(0,np.shape(PBL)[2]):
                cost = abs(ZL[i,:,j,k].squeeze() - PBL[i,j,k])
                index_pbl = np.argmin(cost)
                PBLH[i,j,k] = p_mid[i,index_pbl,j,k]

    # shape up the ctm class
    mind_data = ctm_model(latitude, longitude, time,
                          NO2, HCHO, p_mid, DELP, PBLH, ctmtype)
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
        glob.glob(product_dir + "/*tavg3_3d_gmi_Nv." + str(YYYYMM) + "*.nc4"))
    # define gas profiles for saving
    outputs = []
    for file in tavg3_3d_gas_files:
        outputs.append(minds_reader_core(file))

    return outputs


def colocate_est_conversion(ctmdata, airdata):
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
    # define the triangulation
    points = np.zeros((np.size(ctmdata[0].latitude), 2))
    points[:, 0] = ctmdata[0].longitude.flatten()
    points[:, 1] = ctmdata[0].latitude.flatten()
    tri = Delaunay(points)
    
    time_aircraft = airdata.time.year*10000 + airdata.time.month*100 +\
            airdata.time.day + airdata.time.hour/24.0 + airdata.time.minute / \
            60.0/24.0 + airdata.time.second/3600.0/24.0

    # translating ctm data into aircraft location/time
    ctm_mapped_pressure = []
    ctm_mapped_deltap = []
    ctm_mapped_no2 = []
    ctm_mapped_hcho = []
    ctm_mapped_pblh = []
    for t1 in range(0,np.size(airdata.time)):
        # find the closest day
        closest_index = np.argmin(np.abs(time_aircraft[t1] - time_ctm))
        # find the closest hour (this only works for 3-hourly frequency)
        closest_index_day = int(np.floor(closest_index/8.0))
        closest_index_hour = int(closest_index % 8)
        print("The closest MINDS file used for the aircraft at " + str(airdata.time[t1]) +
              " is at " + str(time_ctm_datetype[closest_index_day][closest_index_hour]))
        # time
        ctm_mid_pressure = ctmdata[closest_index_day].pressure_mid[closest_index_hour, :, :, :].squeeze(
        )
        ctm_deltap = ctmdata[closest_index_day].deltap[closest_index_hour, :, :, :].squeeze(
        )
        ctm_no2_profile = ctmdata[closest_index_day].gas_profile_no2[closest_index_hour, :, :].squeeze(
        )
        ctm_hcho_profile = ctmdata[closest_index_day].gas_profile_hcho[closest_index_hour, :, :].squeeze(
        )
        ctm_PBLH = ctmdata[closest_index_day].PBLH[closest_index_hour, :, :].squeeze(
        )


        ctm_mid_pressure_new = np.zeros((np.shape(ctm_mid_pressure)[0]))*np.nan
        ctm_deltap_new = np.zeros((np.shape(ctm_mid_pressure)[0]))*np.nan
        ctm_no2_profile_new = np.zeros((np.shape(ctm_mid_pressure)[0]))*np.nan
        ctm_hcho_profile_new = np.zeros((np.shape(ctm_mid_pressure)[0]))*np.nan

        air_coordinate = {}
        air_coordinate["Longitude"] = airdata.lon[t1]
        air_coordinate["Latitude"] = airdata.lat[t1]
        # calculate distance to remove too-far estimates
        tree = cKDTree(points)
        grid = np.zeros((2, np.size(air_coordinate["Longitude"])))
        grid[0, :] = air_coordinate["Longitude"]
        grid[1, :] = air_coordinate["Latitude"]
        xi = _ndim_coords_from_arrays(tuple(grid), ndim=points.shape[1])
        dists, _ = tree.query(xi)
        for z in range(0, np.shape(ctm_mid_pressure)[0]):
            ctm_mid_pressure_new[z] = _interpolosis(tri, ctm_mid_pressure[z, :, :].squeeze(), air_coordinate["Longitude"],
                                                          air_coordinate["Latitude"], 2, dists, 0.2)
            ctm_deltap_new[z] = _interpolosis(tri, ctm_deltap[z, :, :].squeeze(), air_coordinate["Longitude"],
                                                          air_coordinate["Latitude"], 2, dists, 0.2)
            ctm_no2_profile_new[z] = _interpolosis(tri, ctm_no2_profile[z, :, :].squeeze(), air_coordinate["Longitude"],
                                                          air_coordinate["Latitude"], 2, dists, 0.2)
            ctm_hcho_profile_new[z] = _interpolosis(tri, ctm_hcho_profile[z, :, :].squeeze(), air_coordinate["Longitude"],
                                                          air_coordinate["Latitude"], 2, dists, 0.2)           

        ctm_PBLH_new = _interpolosis(tri, ctm_PBLH, air_coordinate["Longitude"],
                                           air_coordinate["Latitude"], 2, dists, 0.2)
        
        ctm_mapped_pressure.append(ctm_mid_pressure_new)
        ctm_mapped_deltap.append(ctm_deltap_new)
        ctm_mapped_pblh.append(ctm_PBLH_new)
        ctm_mapped_no2.append(ctm_no2_profile_new)
        ctm_mapped_hcho.append(ctm_hcho_profile_new)

    # converting the lists to numpy array
    ctm_mapped_pressure = np.array(ctm_mapped_pressure)
    ctm_deltap_new = np.array(ctm_deltap_new)
    ctm_mapped_pblh = np.array(ctm_mapped_pblh)
    ctm_mapped_no2 = np.array(ctm_mapped_no2)
    ctm_mapped_hcho = np.array(ctm_mapped_hcho)

    # now colocating based on each spiral number
    unique = np.unique(airdata.profile_num)
    output = {}
    output["HCHO_profile_aircraft"] = []
    output["NO2_profile_aircraft"] = []
    output["pressure"] = []
    output["PBLH"] = []
    output["Spiral_num"] = []
    output["deltap"] = []
    
    for u in range(0, np.size(unique)):
        if unique[u] == 0:
            continue
        aircraft_pres_unique = airdata.altp[np.where(airdata.profile_num == unique[u])]
        aircraft_HCHO_unique = airdata.HCHO_ppbv[np.where(airdata.profile_num == unique[u])]
        aircraft_NO2_unique = airdata.NO2_ppbv[np.where(airdata.profile_num == unique[u])]

        ctm_press_chosen = ctm_mapped_pressure[:,np.where(airdata.profile_num == unique[u])]
        ctm_deltap_chosen = ctm_mapped_deltap[:,np.where(airdata.profile_num == unique[u])]
        # interpolate aircraft vertical grid into minds
        f = interpolate.interp1d(
                np.log(aircraft_pres_unique),
                aircraft_HCHO_unique, fill_value=np.nan)

        interpolated_HCHO = f(np.log(ctm_press_chosen))
        output["HCHO_profile_aircraft"].append(interpolated_HCHO)
        f = interpolate.interp1d(
                np.log(aircraft_pres_unique),
                aircraft_NO2_unique, fill_value=np.nan)
        
        interpolated_NO2 = f(np.log(ctm_press_chosen))
        output["NO2_profile_aircraft"].append(interpolated_NO2)
        output["pressure"].append(ctm_press_chosen)
        output["deltap"].append(ctm_deltap_chosen)
        output["Spiral_num"].append(unique[u])
    
    savemat("discover_aq_minds.mat", output)



if __name__ == "__main__":
    aircraft_path = '/media/asouri/Amir_5TB/NASA/ACMAP_souri_ozonerates/Box_Model/Data/discoveraq-mrg15-p3b_merge_20130904_R3_thru20130929.ict'
    aircraft_data1 = aircraft_reader(aircraft_path, 2013)
    minds_data = minds_reader(folder,'201309')
    colocate_est_conversion(aircraft_data1,minds_data)