from ozonerates.reader import readers
from ozonerates.tools import  write_to_nc, ctmpost, write_to_nc_product
from ozonerates.po3_lasso import PO3est_empirical
from ozonerates.po3_dnn import PO3est_DNN
from ozonerates.report import report
from pathlib import Path

class ozonerates(object):

    def __init__(self) -> None:
        pass

    def read_data(self, ctm_type: str, ctm_freq:str, sensor: str, ctm_path: Path, sat_path: list, YYYYMM: str, output_folder='diag', read_ak=False, trop=True, num_job=1):
        '''
           This function reads GMI and OMI/TROPOMI NO2 and HCHO data and generate ncfiles at the end
        '''
        reader_obj = readers()
        # initialize
        reader_obj.add_ctm_data(ctm_type, ctm_freq, ctm_path)
        reader_obj.read_ctm_data(YYYYMM, num_job=num_job)
        self.ctmdata = reader_obj.ctm_data
        # NO2
        reader_obj.add_satellite_data(
            sensor + '_NO2', sat_path[0])
        reader_obj.read_satellite_data(
            YYYYMM, read_ak=read_ak, trop=trop, num_job=num_job)
        self.satno2 = reader_obj.sat_data
        self.o3paramno2 = ctmpost(self.satno2,self.ctmdata,ctm_freq)
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
            sensor + '_HCHO', sat_path[1])
        reader_obj.read_satellite_data(
            YYYYMM, read_ak=read_ak, trop=trop, num_job=num_job)
        self.sathcho = reader_obj.sat_data
        self.o3paramhcho = ctmpost(self.sathcho,self.ctmdata,ctm_freq)
        self.sathcho = []
        self.ctmdata = []
        reader_obj = []
        # saving as netcdf files
        for fhcho in self.o3paramhcho:
            time_hcho = fhcho.time
            time_hcho = time_hcho.strftime("%Y%m%d_%H%M%S")
            write_to_nc(fhcho, "PO3inputs_FORM_" +
                        str(time_hcho), output_folder)

    def po3estimate_empirical(self, no2_path, hcho_path, startdate, enddate, num_job = 1):
        '''
           Forward estimation of PO3 using LASSO regression output (lasso_piecewise_4group.mat)
        '''
        self.PO3_output_empirical = PO3est_empirical(no2_path, hcho_path, startdate, enddate, num_job = num_job)

    def po3estimate_dnn(self, no2_path, hcho_path, startdate, enddate, num_job = 1):
        '''
           Forward estimation of PO3 using DNN
        '''

        self.PO3_output_dnn = PO3est_DNN(no2_path, hcho_path, startdate, enddate, num_job = num_job)

    def reporting(self, fname: str, folder='report'):
        '''
           Making pdf reports
        '''
        report(self.PO3_output_dnn, fname, folder)

    def writenc(self, fname: str, folder='diag'):
        '''
           Making nc diags
        '''
        write_to_nc_product(self.PO3_output_dnn,fname, folder)


# testing
if __name__ == "__main__":

    ozonerates_obj = ozonerates()
    sat_path = []
    sat_path.append(
        Path('/discover/nobackup/asouri/PROJECTS/PO3_ACMAP/omi_no2_PO3'))
    sat_path.append(
        Path('/discover/nobackup/asouri/PROJECTS/PO3_ACMAP/omi_hcho_PO3'))
    #ozonerates_obj.read_data('GMI', Path('/discover/nobackup/asouri/GITS/OI-SAT-GMI/oisatgmi/download_bucket/gmi/'),
    #                         sat_path, '200506', read_ak=False, trop=True, num_job=12)
    ozonerates_obj.po3estimate_empirical(
        "./diag", "./diag", '2005-06-01', '2005-06-30')
    ozonerates_obj.reporting("PO3_estimates.pdf")
