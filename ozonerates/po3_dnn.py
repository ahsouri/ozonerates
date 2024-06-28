import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error, mean_squared_error
import warnings
import sys
import pickle
from scipy.stats import gaussian_kde
import scipy.io as sio
import numpy as np
import glob
import warnings
from netCDF4 import Dataset
from scipy.interpolate import interpn
import datetime
from ozonerates.config import param_output
from joblib import Parallel, delayed

warnings.filterwarnings("ignore",category=RuntimeWarning)

def _daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


def _read_nc(filename, var):
    # reading nc files without a group
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    out = np.array(nc_fid.variables[var])
    nc_fid.close()
    return np.squeeze(out)

def PO3est_DNN(no2_path, hcho_path, startdate, enddate, num_job=1):
    '''
       Forward estimation of PO3 based on information from MERRA2GMI and OMI/TROPOMI
       The output will be on daily basis
    '''

    # converting the string time to datetime format
    start_date = datetime.date(int(startdate[0:4]), int(
        startdate[5:7]), int(startdate[8:10]))
    end_date = datetime.date(int(enddate[0:4]), int(
        enddate[5:7]), int(enddate[8:10]))

    PO3_estimates = []
    inputs = {}
    inputs["H2O"] = []
    inputs["FNR"] = []
    inputs["J1"] = []
    inputs["J4"] = []
    inputs["HCHO_ppbv"] = []
    inputs["NO2_ppbv"] = []
    inputs["SJ4"] = []
    inputs["SJ1"] = []
    inputs["SHCHO"] = []
    inputs["SNO2"] = []
    inputs["SH2O"] = []
    inputs["VCD_NO2"] = []
    inputs["VCD_FORM"] = []
    inputs["PBL_no2_factor"] = []
    inputs["PBL_form_factor"] = []
    inputs["PO3_err"] = []

    for single_date in _daterange(start_date, end_date):

        no2_files = sorted((glob.glob(no2_path + "/*_NO2_" + str(single_date.year) + f"{single_date.month:02}"
                                      + f"{single_date.day:02}" + "*.nc")))
        hcho_files = sorted((glob.glob(hcho_path + "/*_FORM_" + str(single_date.year) + f"{single_date.month:02}"
                                       + f"{single_date.day:02}" + "*.nc")))
        # we make a list of inputs to append and average later for diags
        PBLH = []
        VCD_NO2 = []
        VCD_FORM = []
        PBL_no2_factor = []
        PBL_form_factor = []
        PL = []
        T = []
        surface_albedo_no2 = []
        O3col = []
        SZA = []
        surface_alt = []
        VCD_NO2_err = []
        VCD_HCHO_err = []
        H2O = []
        # reading NO2 files daily
        for f in no2_files:
            print(f)
            PBLH.append(_read_nc(f, 'PBLH'))
            VCD_NO2.append(_read_nc(f, 'VCD'))
            PBL_no2_factor.append(_read_nc(f, 'gas_pbl_factor_no2'))
            PL.append(_read_nc(f, 'pressure_mid'))
            T.append(_read_nc(f, 'temperature_mid'))
            surface_albedo_no2.append(_read_nc(f, 'surface_albedo'))
            O3col.append(_read_nc(f, 'O3col'))
            SZA.append(_read_nc(f, 'SZA'))
            surface_alt.append(_read_nc(f, 'surface_alt'))
            latitude = _read_nc(f, 'latitude')
            longitude = _read_nc(f, 'longitude')
            VCD_NO2_err.append(_read_nc(f, 'VCD_err'))
            H2O.append(_read_nc(f,'H2O'))
        # reading FORM files daily
        for f in hcho_files:
            VCD_FORM.append(_read_nc(f, 'VCD'))
            PBL_form_factor.append(_read_nc(f, 'gas_pbl_factor_hcho'))
            VCD_HCHO_err.append(_read_nc(f, 'VCD_err'))

        # averaging to make daily coverage from L3 swaths
        PBLH = np.nanmean(np.array(PBLH), axis=0)
        VCD_NO2 = np.nanmean(np.array(VCD_NO2), axis=0)
        VCD_FORM = np.nanmean(np.array(VCD_FORM), axis=0)
        PBL_no2_factor = np.nanmean(np.array(PBL_no2_factor), axis=0)
        PBL_form_factor = np.nanmean(np.array(PBL_form_factor), axis=0)
        PL = np.nanmean(np.array(PL), axis=0)
        T = np.nanmean(np.array(T), axis=0)
        H2O = np.nanmean(np.array(H2O), axis=0)
        surface_albedo_no2 = np.nanmean(np.array(surface_albedo_no2), axis=0)
        O3col = np.nanmean(np.array(O3col), axis=0)
        SZA = np.nanmean(np.array(SZA), axis=0)
        surface_alt = np.nanmean(np.array(surface_alt), axis=0)
        VCD_HCHO_err = np.sqrt(np.nanmean((np.array(VCD_HCHO_err))**2, axis=0))
        VCD_NO2_err = np.sqrt(np.nanmean((np.array(VCD_NO2_err))**2, axis=0))
        # oceanic areas sometimes are negative
        surface_alt[surface_alt <= 0] = 0.0
        # extract the features: H2O, HCHO_ppbv, NO2_ppbv, jNO2, jO1D

        NO2_ppbv = VCD_NO2*PBL_no2_factor
        HCHO_ppbv = VCD_FORM*PBL_form_factor
        NO2_ppbv_err = np.sqrt((VCD_NO2_err*PBL_no2_factor)**2 +
                               (0.01*VCD_NO2*PBL_no2_factor)**2 + (0.32*PBL_no2_factor)**2)
        HCHO_ppbv_err = np.sqrt((VCD_HCHO_err*PBL_form_factor)**2 +
                                (0.01*VCD_FORM*PBL_form_factor)**2 + (0.90*PBL_form_factor)**2)

        # extrating J values from a LUT
        Jvalue = sio.loadmat('../data/HybridJtables.mat')
        SZAhybrid = Jvalue["SZAhybrid"][:, 0, 0, 0]
        ALBhybrid = Jvalue["ALBhybrid"][0, :, 0, 0]
        O3Chybrid = Jvalue["O3Chybrid"][0, 0, :, 0]
        ALThybrid = Jvalue["ALThybrid"][0, 0, 0, :]
        J = Jvalue["Jhybrid"]
        # J4 is NO2+hv-> NO + O
        J4 = (J["J4"])
        J4 = np.array(J4[0, 0])

        # J1 is O3 + hv -> O1D
        J1 = (J["J1"])
        J1 = np.array(J1[0, 0])
        # linear interpolation (extrapolation is not allowed = NaN)
        J4 = interpn((SZAhybrid.flatten(), ALBhybrid.flatten(), O3Chybrid.flatten(), ALThybrid.flatten()),
                     J4, (SZA.flatten(), surface_albedo_no2.flatten(),
                          O3col.flatten(), surface_alt.flatten()),
                     method="linear", bounds_error=False, fill_value=np.nan)
        J4 = np.reshape(J4, (np.shape(NO2_ppbv)[0], np.shape(NO2_ppbv)[1]))
        J1 = interpn((SZAhybrid.flatten(), ALBhybrid.flatten(), O3Chybrid.flatten(), ALThybrid.flatten()),
                     J1, (SZA.flatten(), surface_albedo_no2.flatten(),
                          O3col.flatten(), surface_alt.flatten()),
                     method="linear", bounds_error=False, fill_value=np.nan)
        J1 = np.reshape(J1, (np.shape(NO2_ppbv)[0], np.shape(NO2_ppbv)[1]))

        # load the DNN model
        dnn_model = keras.models.load_model('../data/all_best_model.keras')

        normalization_factors = [0.1,3e-5,1.0,1e2,1e1] # jNO2, jO1D, H2O, NO2, HCHO

        inputs_dnn = np.zeros((np.size(J1),5))
        inputs_dnn[:,0] = J4.flatten()/normalization_factors[0]
        inputs_dnn[:,1] = J1.flatten()/normalization_factors[1]
        inputs_dnn[:,2] = H2O.flatten()
        inputs_dnn[:,3] = NO2_ppbv.flatten()/normalization_factors[3]
        inputs_dnn[:,4] = HCHO_ppbv.flatten()/normalization_factors[4]

        prediction = np.array(dnn_model.predict(inputs_dnn,verbose=1))
        PO3 = np.reshape(prediction, (np.shape(NO2_ppbv)[0], np.shape(NO2_ppbv)[1]))

        #SNO2
        inputs_dnn_sens = inputs_dnn
        inputs_dnn_sens[:,3] = inputs_dnn[:,3]*1.1
        prediction_up = np.array(dnn_model.predict(inputs_dnn_sens,verbose=1))

        inputs_dnn_sens = inputs_dnn
        inputs_dnn_sens[:,3] = inputs_dnn[:,3]*0.9
        prediction_down = np.array(dnn_model.predict(inputs_dnn_sens,verbose=1))

        SNO2 = (prediction_up - prediction_down)/0.2
        SNO2 = np.reshape(SNO2, (np.shape(NO2_ppbv)[0], np.shape(NO2_ppbv)[1]))

        #SHCHO
        inputs_dnn_sens = inputs_dnn
        inputs_dnn_sens[:,4] = inputs_dnn[:,4]*1.1
        prediction_up = np.array(dnn_model.predict(inputs_dnn_sens,verbose=1))

        inputs_dnn_sens = inputs_dnn
        inputs_dnn_sens[:,4] = inputs_dnn[:,4]*0.9
        prediction_down = np.array(dnn_model.predict(inputs_dnn_sens,verbose=1))

        SHCHO = (prediction_up - prediction_down)/0.2
        SHCHO = np.reshape(SHCHO, (np.shape(NO2_ppbv)[0], np.shape(NO2_ppbv)[1]))

        #SJ1
        inputs_dnn_sens = inputs_dnn
        inputs_dnn_sens[:,1] = inputs_dnn[:,1]*1.1
        prediction_up = np.array(dnn_model.predict(inputs_dnn_sens,verbose=1))

        inputs_dnn_sens = inputs_dnn
        inputs_dnn_sens[:,1] = inputs_dnn[:,1]*0.9
        prediction_down = np.array(dnn_model.predict(inputs_dnn_sens,verbose=1))

        SJ1 = (prediction_up - prediction_down)/0.2
        SJ1 = np.reshape(SJ1, (np.shape(NO2_ppbv)[0], np.shape(NO2_ppbv)[1]))

        #SJ4
        inputs_dnn_sens = inputs_dnn
        inputs_dnn_sens[:,0] = inputs_dnn[:,0]*1.1
        prediction_up = np.array(dnn_model.predict(inputs_dnn_sens,verbose=1))

        inputs_dnn_sens = inputs_dnn
        inputs_dnn_sens[:,0] = inputs_dnn[:,0]*0.9
        prediction_down = np.array(dnn_model.predict(inputs_dnn_sens,verbose=1))

        SJ4 = (prediction_up - prediction_down)/0.2
        SJ4 = np.reshape(SJ4, (np.shape(NO2_ppbv)[0], np.shape(NO2_ppbv)[1]))

        #SH2O
        inputs_dnn_sens = inputs_dnn
        inputs_dnn_sens[:,2] = inputs_dnn[:,2]*1.1
        prediction_up = np.array(dnn_model.predict(inputs_dnn_sens,verbose=1))

        inputs_dnn_sens = inputs_dnn
        inputs_dnn_sens[:,2] = inputs_dnn[:,2]*0.9
        prediction_down = np.array(dnn_model.predict(inputs_dnn_sens,verbose=1))

        SH2O = (prediction_up - prediction_down)/0.2
        SH2O = np.reshape(SH2O, (np.shape(NO2_ppbv)[0], np.shape(NO2_ppbv)[1]))

        # append inputs and PO3_estimates daily
        PO3_estimates.append(PO3)
        inputs["FNR"].append(HCHO_ppbv/NO2_ppbv)
        inputs["H2O"].append(H2O)
        inputs["J1"].append(J1*1e6)
        inputs["J4"].append(J4*1e3)
        inputs["HCHO_ppbv"].append(HCHO_ppbv)
        inputs["NO2_ppbv"].append(NO2_ppbv)
        inputs["SJ4"].append(SJ4)
        inputs["SJ1"].append(SJ1)
        inputs["SH2O"].append(SH2O)
        inputs["SHCHO"].append(SHCHO)
        inputs["SNO2"].append(SNO2)
        inputs["VCD_NO2"].append(VCD_NO2)
        inputs["VCD_FORM"].append(VCD_FORM)
        inputs["PBL_no2_factor"].append(PBL_no2_factor)
        inputs["PBL_form_factor"].append(PBL_form_factor)
        inputs["PO3_err"].append(PO3*0.0)

    FNR = np.array(inputs["FNR"])
    H2O = np.array(inputs["H2O"])
    J1 = np.array(inputs["J1"])
    J4 = np.array(inputs["J4"])
    HCHO_ppbv = np.array(inputs["HCHO_ppbv"])
    NO2_ppbv = np.array(inputs["NO2_ppbv"])
    J1_contrib = np.array(inputs["SJ1"])
    J4_contrib = np.array(inputs["SJ4"])
    NO2_contrib = np.array(inputs["SNO2"])
    HCHO_contrib = np.array(inputs["SHCHO"])
    SH2O = np.array(inputs["SH2O"])
    VCD_NO2 = np.array(inputs["VCD_NO2"])
    VCD_FORM = np.array(inputs["VCD_FORM"])
    PBL_no2_factor = np.array(inputs["PBL_no2_factor"])
    PBL_form_factor = np.array(inputs["PBL_form_factor"])
    PO3_estimates = np.array(PO3_estimates)
    PO3_err = np.array(inputs["PO3_err"])

    output = param_output(latitude, longitude, VCD_NO2, PBL_no2_factor, VCD_FORM, PBL_form_factor, PO3_estimates,
                          FNR, H2O, HCHO_ppbv, NO2_ppbv, J4, J1, HCHO_contrib, NO2_contrib, J4_contrib, J1_contrib, SH2O,PO3_err)
    return output
