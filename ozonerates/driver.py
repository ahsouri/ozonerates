from ozonerates.reader import readers
from pathlib import Path
import numpy as np
from scipy.io import savemat
from numpy import dtype
from netCDF4 import Dataset
import os


class ozonerates(object):

    def __init__(self) -> None:
        pass

    def read_data(self, ctm_type: str, ctm_path: Path,
                  sat_type: str, sat_path: Path, YYYYMM: str, read_ak=False, trop=True, num_job=1):
        reader_obj = readers()
        #initialize
        reader_obj.add_ctm_data(ctm_type, ctm_path)
        reader_obj.read_ctm_data(YYYYMM, num_job=num_job)
        self.ctmdata = reader_obj.ctm_data
        # NO2
        reader_obj.add_satellite_data(
            'OMI_NO2', sat_path)
        reader_obj.read_satellite_data(
            YYYYMM, read_ak=read_ak, trop=trop, num_job=num_job)
        self.satno2 = reader_obj.sat_data
        
        # HCHO
        reader_obj.add_satellite_data(
            'OMI_HCHO', sat_path)
        reader_obj.read_satellite_data(
            YYYYMM, read_ak=read_ak, trop=trop, num_job=num_job)

        self.sathcho = reader_obj.sat_data
        reader_obj = []

    def average(self, startdate: str, enddate: str, gasname=None):
        '''
            average the data
            Input:
                startdate [str]: starting date in YYYY-mm-dd format string
                enddate [str]: ending date in YYYY-mm-dd format string  
        '''


    def reporting(self, fname: str, gasname, folder='report'):
        pass
 
    def write_to_nc(self, output_file, output_folder='diag'):
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
        ncfile.createDimension('x', np.shape(self.sat_averaged_vcd)[0])
        ncfile.createDimension('y', np.shape(self.sat_averaged_vcd)[1])

        data1 = ncfile.createVariable(
            'sat_averaged_vcd', dtype('float32').char, ('x', 'y'))
        data1[:, :] = self.sat_averaged_vcd

        data2 = ncfile.createVariable(
            'ctm_averaged_vcd_prior', dtype('float32').char, ('x', 'y'))
        data2[:, :] = self.ctm_averaged_vcd

        data3 = ncfile.createVariable(
            'ctm_averaged_vcd_posterior', dtype('float32').char, ('x', 'y'))
        data3[:, :] = self.ctm_averaged_vcd_corrected

        data4 = ncfile.createVariable(
            'sat_averaged_error', dtype('float32').char, ('x', 'y'))
        data4[:, :] = self.sat_averaged_error

        data5 = ncfile.createVariable(
            'ak_OI', dtype('float32').char, ('x', 'y'))
        data5[:, :] = self.ak_OI

        data6 = ncfile.createVariable(
            'error_OI', dtype('float32').char, ('x', 'y'))
        data6[:, :] = self.error_OI

        scaling_factor = self.ctm_averaged_vcd_corrected/self.ctm_averaged_vcd
        scaling_factor[np.where((np.isnan(scaling_factor)) | (np.isinf(scaling_factor)) |
                       (scaling_factor == 0.0))] = 1.0
        data7 = ncfile.createVariable(
            'scaling_factor', dtype('float32').char, ('x', 'y'))
        data7[:, :] = scaling_factor

        data8 = ncfile.createVariable(
            'lon', dtype('float32').char, ('x', 'y'))
        data8[:, :] = self.reader_obj.sat_data[0].longitude_center

        data9 = ncfile.createVariable(
            'lat', dtype('float32').char, ('x', 'y'))
        data9[:, :] = self.reader_obj.sat_data[0].latitude_center

        data10 = ncfile.createVariable(
            'aux1', dtype('float32').char, ('x', 'y'))
        data10[:, :] = self.aux1

        data11 = ncfile.createVariable(
            'aux2', dtype('float32').char, ('x', 'y'))
        data11[:, :] = self.aux2

        ncfile.close()


# testing
if __name__ == "__main__":

    ozonerates_obj = ozonerates()
    oisatgmi_obj.read_data('GMI', Path('/home/asouri/git_repos/mule/eccoh_sample'), 'NO2', 'monthly', 'OMI_NO2',
                           Path('download_bucket/omi_no2/'), '201803',
                           averaged=True, read_ak=False, trop=True, num_job=1)
    oisatgmi_obj.recal_amf()
    #oisatgmi_obj.conv_ak()
    oisatgmi_obj.average('2018-03-01', '2018-04-01')
    oisatgmi_obj.oi(error_ctm=10.0)
    oisatgmi_obj.reporting('NO2_200503_new', 'NO2', folder='report')
    oisatgmi_obj.write_to_nc('NO2_200503_new', 'diag')
