# ------------------------------------- #
# Ozonerates model results writer class #
# using NETCDF4 format suitable for ing #
# estion in NASA's archiving system.    #
# Uses the CF standard regarding the de #
# finition of dimensions, variables and #
# attributes.                           #
#                                       #
# Gonzalo Gonzalez Abad                 #
# ggonzalezabad@cfa.harvard.edu         #
# December 2024                         #
# ------------------------------------- #

# Import needed packages
import warnings
import numpy as np
from netCDF4 import Dataset
from datetime import datetime
from cftime import date2num
from time import asctime, gmtime, strftime
from sys import exit

warnings.filterwarnings('ignore')

def log_message(msg,error=False):
    '''Print message with time stamp
    ARGS:
        msg(string): Message to be printed
        error(bool,optional): If true raise Error to calling program
    '''
    if error:
        m = '{0}: ERROR!!! {1}'.format(asctime(),msg)
        exit(m)
    else:
        m = '{0}: {1}'.format(asctime(),msg)
        print(m)
        
class ozonerate_netcdf_writer:
    '''
        # ------------------------------------- #
        # Ozonerates model results writer class #
        # using NETCDF4 format suitable for ing #
        # estion in NASA's archiving system.    #
        # Uses the CF standard regarding the de #
        # finition of dimensions, variables and #
        # attributes.                           #
        #                                       #
        # Gonzalo Gonzalez Abad                 #
        # ggonzalezabad@cfa.harvard.edu         #
        # December 2024                         #
        # ------------------------------------- #
    '''

    def __init__(self,filename):
        ''' Initialize ozonerates netCDF4 writer
            ARGS:
                filename(string): Filename (or full path plus filename)
                                  of the ozonerates output file
            RETURNS:
                None
        '''
        self.filename = filename
        self.ncid = None
    
    def __enter__(self):
        ''' Open and create netCDF4 file
            ARGS:
                None
            RETURNS:
                self class object
        '''
        self.ncid = Dataset(self.filename,'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        ''' Close and capture exceptions if needed
            ARGS:
                None
            RETURNS
                None
        '''
        if self.ncid:
            self.ncid.close()
    
    def create_dimension(self,name,length):
        ''' Create dimension in netCDF4 file
            ARGS:
                name(string): dimension name
                length(integer): dimension length
            RETURNS:
                None
        '''
        try:
            log_message('creating netCFDF4 dimension {}: {}'.format(name,length))
            dummy = self.ncid.createDimension(name,length)
        except Exception as e:
            log_message('error creating dimension {}: {}'.format(name,length))
            log_message(e,error=True)

    def create_global_attributes(self,attributes):
        ''' Create global attributes in netCDF4 file
            ARGS:
                attributes(dictionary): Global attributes
            RETURNS:
                none
        '''
        try:
            log_message('writing global attributes')
            self.ncid.setncatts(attributes)
        except Exception as e:
            log_message('error writing global attributes')
            log_message(e,error=True)

    def create_variable(self,name='variable',datatype=np.float32,dimensions=None,compression='zlib',complevel=1,
                        shuffle=True,fill_value=-9999.0,values=1,least_significant_digig=None,attributes={}):
        ''' Create variable name in netCDF4 file root
            ARGS:
                name(string): Variable name
                datatype(np data type): Data type of the variable. Default: np.float32
                dimensions(tupple of strings): Tupple of defined variable strings
                compression(string): Default: zlib
                complevel(integer): Value between 0 and 9. Default: 1
                shuffle(bool): Shuffle to improve compression. Default: True
                fill_value(scalar): Default: -9999
                values(scalar of array): Scarlar of array with the values to be saved.
                least_significant_digit(integer): 1 -> 0.1; 2->0.2 ...
                attributes(dictionary): Dictionary of variable attributes
        '''
        try:
            log_message('saving variable {}'.format(name))
            var = self.ncid.createVariable(name,datatype,dimensions,compression=compression,complevel=complevel,
                                           shuffle=shuffle,fill_value=fill_value,
                                           least_significant_digit=least_significant_digig)
            var[:] = values
            for n,v in attributes.items():
                var.setncattr(n,v)
        except Exception as e:
            log_message('error writing {}'.format(name))
            log_message(e,error=True)


    