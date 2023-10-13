from ozonerates.reader import readers
from ozonerates.tools import  write_to_nc
from ozonerates.po3_est import PO3est_empirical
from pathlib import Path
from scipy.spatial import Delaunay
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
from scipy.spatial import cKDTree
from ozonerates.interpolator import _interpolosis
from ozonerates.config import param_input
import numpy as np

class ozonerates(object):

    def __init__(self) -> None:
        pass

    def read_data(self, ctm_type: str, ctm_path: Path, sat_path: list, YYYYMM: str, output_folder='diag', read_ak=False, trop=True, num_job=1):
        '''
           This function reads GMI and OMI/TROPOMI NO2 and HCHO data and generate ncfiles at the end
        '''
        reader_obj = readers()
        # initialize
        reader_obj.add_ctm_data(ctm_type, ctm_path)
        reader_obj.read_ctm_data(YYYYMM, num_job=num_job)
        self.ctmdata = reader_obj.ctm_data
        # NO2
        reader_obj.add_satellite_data(
            'OMI_NO2', sat_path[0])
        reader_obj.read_satellite_data(
            YYYYMM, read_ak=read_ak, trop=trop, num_job=num_job)
        self.satno2 = reader_obj.sat_data
        self.o3paramno2 = self.ctmpost(1)
        self.satno2 = []
        reader_obj.sat_data = []
        # saving as netcdf files
        for fno2 in self.o3paramno2:
            time_no2 = fno2.time
            time_no2 = time_no2.strftime("%Y%m%d_%H%M%S")
            write_to_nc(fno2, "PO3inputs_NO2_" +
                        str(time_no2), output_folder)
        self.o3paramno2 = []
        # HCHO
        reader_obj.add_satellite_data(
            'OMI_HCHO', sat_path[1])
        reader_obj.read_satellite_data(
            YYYYMM, read_ak=read_ak, trop=trop, num_job=num_job)
        self.sathcho = reader_obj.sat_data
        self.o3paramhcho = self.ctmpost(2)
        self.sathcho = []
        self.ctmdata = []
        reader_obj = []
        # saving as netcdf files
        for fhcho in self.o3paramhcho:
            time_hcho = fhcho.time
            time_hcho = time_hcho.strftime("%Y%m%d_%H%M%S")
            write_to_nc(fhcho, "PO3inputs_FORM_" +
                        str(time_hcho), output_folder)

    def po3estimate_empirical(self, no2_path, hcho_path, startdate, enddate):
        '''
           Forward estimation of PO3 using LASSO regression output (lasso_piecewise.mat)
        '''
        PO3est_empirical(no2_path, hcho_path, startdate, enddate)

    def reporting(self, fname: str, gasname, folder='report'):
        pass

    def ctmpost(self,switch):
        print('Mapping CTM data into Sat grid...')
        # list the time in ctm_data
        time_ctm = []
        time_ctm_datetype = []
        for ctm_granule in self.ctmdata:
            time_temp = ctm_granule.time
            for n in range(len(time_temp)):
                time_temp2 = time_temp[n].year*10000 + time_temp[n].month*100 +\
                    time_temp[n].day + time_temp[n].hour/24.0 + \
                    time_temp[n].minute/60.0/24.0 + time_temp[n].second/3600.0/24.0
                time_ctm.append(time_temp2)
            time_ctm_datetype.append(ctm_granule.time)
        time_ctm = np.array(time_ctm)
        # define the triangulation
        points = np.zeros((np.size(self.ctmdata[0].latitude), 2))
        points[:, 0] = self.ctmdata[0].longitude.flatten()
        points[:, 1] = self.ctmdata[0].latitude.flatten()
        tri = Delaunay(points)
        # loop over the satellite list
        counter = 0
        params = []
        if switch==1:
           var=self.satno2
           self.satno2 = []
        if switch==2: 
           var=self.sathcho
           self.sathcho = []
        for L2_granule in var:
            if (L2_granule is None):
                counter += 1
                continue
            time_sat_datetime = L2_granule.time
            time_sat = time_sat_datetime.year*10000 + time_sat_datetime.month*100 +\
                time_sat_datetime.day + time_sat_datetime.hour/24.0 + time_sat_datetime.minute / \
                60.0/24.0 + time_sat_datetime.second/3600.0/24.0
            # find the closest day
            closest_index = np.argmin(np.abs(time_sat - time_ctm))
            # find the closest hour (this only works for 3-hourly frequency)
            closest_index_day = int(np.floor(closest_index/8.0))
            closest_index_hour = int(closest_index % 8)

            print("The closest GMI file used for the L2 at " + str(L2_granule.time) +
                  " is at " + str(time_ctm_datetype[closest_index_day][closest_index_hour]))
            ctm_mid_pressure = self.ctmdata[closest_index_day].pressure_mid[closest_index_hour, :, :, :].squeeze(
            )
            ctm_no2_profile_factor = self.ctmdata[closest_index_day].gas_profile_no2[closest_index_hour, :, :].squeeze(
            )
            ctm_hcho_profile_factor = self.ctmdata[closest_index_day].gas_profile_hcho[closest_index_hour, :, :].squeeze(
            )
            ctm_mid_T = self.ctmdata[closest_index_day].tempeature_mid[closest_index_hour, :, :, :].squeeze(
            )
            ctm_height = self.ctmdata[closest_index_day].height_mid[closest_index_hour, :, :, :].squeeze(
            )
            ctm_O3col = self.ctmdata[closest_index_day].O3col[closest_index_hour, :, :].squeeze(
            )
            ctm_PBLH = self.ctmdata[closest_index_day].PBLH[closest_index_hour, :, :].squeeze(
            )

            ctm_mid_pressure_new = np.zeros((np.shape(ctm_mid_pressure)[0],
                                         np.shape(L2_granule.longitude_center)[0], np.shape(
                                             L2_granule.longitude_center)[1],
                                         ))*np.nan

            ctm_mid_T_new = np.zeros_like(ctm_mid_pressure_new)*np.nan
            ctm_height_new = np.zeros_like(ctm_mid_pressure_new)*np.nan
            ctm_O3col_new = np.zeros((np.shape(L2_granule.longitude_center)[0], np.shape(
                L2_granule.longitude_center)[1],
            ))*np.nan
            ctm_PBLH_new = np.zeros((np.shape(L2_granule.longitude_center)[0], np.shape(
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
            for z in range(0, np.shape(ctm_mid_pressure)[0]):
                ctm_mid_pressure_new[z, :, :] = _interpolosis(tri, ctm_mid_pressure[z, :, :].squeeze(), sat_coordinate["Longitude"],
                                                          sat_coordinate["Latitude"], 1, dists, 0.2)*L2_granule.quality_flag
                ctm_mid_T_new[z, :, :] = _interpolosis(tri, ctm_mid_T[z, :, :].squeeze(), sat_coordinate["Longitude"],
                                                   sat_coordinate["Latitude"], 1, dists, 0.2)*L2_granule.quality_flag
                ctm_height_new[z, :, :] = _interpolosis(tri, ctm_height[z, :, :].squeeze(), sat_coordinate["Longitude"],
                                                    sat_coordinate["Latitude"], 1, dists, 0.2)*L2_granule.quality_flag
            ctm_O3col_new[:, :] = _interpolosis(tri, ctm_O3col, sat_coordinate["Longitude"],
                                            sat_coordinate["Latitude"], 1, dists, 0.2)*L2_granule.quality_flag
            ctm_PBLH_new[:, :] = _interpolosis(tri, ctm_PBLH, sat_coordinate["Longitude"],
                                           sat_coordinate["Latitude"], 1, dists, 0.2)*L2_granule.quality_flag
            ctm_no2_profile_f_new[:, :] = _interpolosis(tri, ctm_no2_profile_factor, sat_coordinate["Longitude"],
                                                    sat_coordinate["Latitude"], 1, dists, 0.2)*L2_granule.quality_flag
            ctm_hcho_profile_f_new[:, :] = _interpolosis(tri, ctm_hcho_profile_factor, sat_coordinate["Longitude"],
                                                     sat_coordinate["Latitude"], 1, dists, 0.2)*L2_granule.quality_flag

            param = param_input(L2_granule.longitude_center, L2_granule.latitude_center, L2_granule.time,
                            ctm_no2_profile_f_new, ctm_hcho_profile_f_new, ctm_O3col_new, ctm_mid_pressure_new,
                            ctm_mid_T_new, ctm_height_new, ctm_PBLH_new, L2_granule.vcd, L2_granule.uncertainty,
                            L2_granule.tropopause, L2_granule.surface_albedo, L2_granule.SZA, L2_granule.surface_alt)
            params.append(param)
        return params

# testing
if __name__ == "__main__":

    ozonerates_obj = ozonerates()
    sat_path = []
    sat_path.append(
        Path('/discover/nobackup/asouri/PROJECTS/PO3_ACMAP/omi_no2_PO3'))
    sat_path.append(
        Path('/discover/nobackup/asouri/PROJECTS/PO3_ACMAP/omi_hcho_PO3'))
    ozonerates_obj.read_data('GMI', Path('/discover/nobackup/asouri/GITS/OI-SAT-GMI/oisatgmi/download_bucket/gmi/'),
                             sat_path, '200506', read_ak=False, trop=True, num_job=24)
    ozonerates_obj.po3estimate_empirical(
        "./diag", "./diag", '2005-06-01', '2005-06-10')
