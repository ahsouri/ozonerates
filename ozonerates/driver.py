from ozonerates.reader import readers
from ozonerates.tools import write_to_nc, ctmpost, write_to_nc_product, tropomi_albedo
from ozonerates.po3_lasso import PO3est_empirical
from ozonerates.po3_dnn import PO3est_DNN
from ozonerates.report import report
from pathlib import Path
from scipy.spatial import Delaunay
from scipy.io import loadmat
import numpy as np


class ozonerates(object):

    def __init__(self) -> None:
        pass

    def read_data(self, ctm_type: str, ctm_freq: str, sensor: str, ctm_path: Path, sat_path: list, YYYYMM: str,
                  output_folder='diag', read_ak=False, trop=True, num_job=1):
        """
        Reads GMI/MINDS and OMI/TROPOMI NO2 and HCHO data and generates NetCDF files.

        Parameters:
        ctm_type (str): Type of CTM data.
        ctm_freq (str): Frequency of CTM data.
        sensor (str): Satellite sensor name.
        ctm_path (Path): Path to CTM data.
        sat_path (list): List of satellite data paths.
        YYYYMM (str): Year and month in YYYYMM format.
        output_folder (str, optional): Folder to save output files. Defaults to 'diag'.
        read_ak (bool, optional): Whether to read averaging kernels. Defaults to False.
        trop (bool, optional): Whether to include tropospheric data. Defaults to True.
        num_job (int, optional): Number of jobs for parallel processing. Defaults to 1.
        """
        reader_obj = readers()

        # Initialize and read CTM data
        reader_obj.add_ctm_data(ctm_type, ctm_freq, ctm_path)
        reader_obj.read_ctm_data(YYYYMM, num_job=num_job)
        self.ctmdata = reader_obj.ctm_data

        # Process NO2 data
        reader_obj.add_satellite_data(f"{sensor}_NO2", sat_path[0])
        reader_obj.read_satellite_data(
            YYYYMM, read_ak=read_ak, trop=trop, num_job=num_job)
        self.satno2 = reader_obj.sat_data
        self.o3paramno2 = ctmpost(self.satno2, self.ctmdata, ctm_freq)

        # Clear temporary data
        self.satno2 = []
        reader_obj.sat_data = []

        # Precompute TROPOMI surface albedo interpolation
        tropomi_data = loadmat('../data/tropomi_ler_uv_vis.mat')
        lat_tropomi = tropomi_data["lat"]
        lon_tropomi = tropomi_data["lon"]

        # Define Delaunay triangulation
        points = np.column_stack((lon_tropomi.flatten(), lat_tropomi.flatten()))
        tri = Delaunay(points)

        # Save NO2 data to NetCDF files
        for fno2 in self.o3paramno2:
            time_no2 = fno2.time.strftime("%Y%m%d_%H%M%S")
            fno2.surface_albedo = tropomi_albedo(
                tri, False, fno2.latitude, fno2.longitude, int(YYYYMM[4:]))
            write_to_nc(fno2, f"PO3inputs_NO2_{time_no2}", output_folder)

        # Clear temporary data
        self.o3paramno2 = []

        # Process HCHO data
        reader_obj.add_satellite_data(f"{sensor}_HCHO", sat_path[1])
        reader_obj.read_satellite_data(
            YYYYMM, read_ak=read_ak, trop=trop, num_job=num_job)
        self.sathcho = reader_obj.sat_data
        self.o3paramhcho = ctmpost(self.sathcho, self.ctmdata, ctm_freq)

        # Clear temporary data
        self.sathcho = []
        self.ctmdata = []
        reader_obj = []

        # Save HCHO data to NetCDF files
        for fhcho in self.o3paramhcho:
            time_hcho = fhcho.time.strftime("%Y%m%d_%H%M%S")
            fhcho.surface_albedo = tropomi_albedo(
                tri, True, fhcho.latitude, fhcho.longitude, int(YYYYMM[4:]))
            write_to_nc(fhcho, f"PO3inputs_FORM_{time_hcho}", output_folder)

    def po3estimate_empirical(self, no2_path, hcho_path, startdate, enddate, num_job=1):
        '''
           Forward estimation of PO3 using LASSO regression output (lasso_piecewise_4group.mat)
        '''
        self.PO3_output = PO3est_empirical(
            no2_path, hcho_path, startdate, enddate, num_job=num_job)

    def po3estimate_dnn(self, no2_path, hcho_path, startdate, enddate, num_job=1):
        '''
           Forward estimation of PO3 using DNN
        '''

        self.PO3_output = PO3est_DNN(
            no2_path, hcho_path, startdate, enddate, num_job=num_job)

    def reporting(self, fname: str, folder='report'):
        '''
           Making pdf reports
        '''
        report(self.PO3_output, fname, folder)

    def writenc(self, fname: str, folder='diag'):
        '''
           Making nc diags
        '''
        write_to_nc_product(self.PO3_output, fname, folder)


# testing
if __name__ == "__main__":

    ozonerates_obj = ozonerates()
    sat_path = []
    sat_path.append(
        Path('/discover/nobackup/asouri/PROJECTS/PO3_ACMAP/omi_no2_PO3'))
    sat_path.append(
        Path('/discover/nobackup/asouri/PROJECTS/PO3_ACMAP/omi_hcho_PO3'))
    # ozonerates_obj.read_data('GMI', Path('/discover/nobackup/asouri/GITS/OI-SAT-GMI/oisatgmi/download_bucket/gmi/'),
    #                         sat_path, '200506', read_ak=False, trop=True, num_job=12)
    ozonerates_obj.po3estimate_empirical(
        "./diag", "./diag", '2005-06-01', '2005-06-30')
    ozonerates_obj.reporting("PO3_estimates.pdf")
