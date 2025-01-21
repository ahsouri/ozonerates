import scipy.io as sio
import numpy as np
import glob
import warnings
from netCDF4 import Dataset
from scipy.interpolate import interpn
import datetime
from ozonerates.config import param_output
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=RuntimeWarning)


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


def loop_estimator(J4_m, J1_m, HCHO_m, NO2_m, HCHO_err_m, NO2_err_m, COEFFs, COEFF0s, n_member):
    s_no2 = np.random.normal(0, NO2_err_m, n_member)
    s_hcho = np.random.normal(0, HCHO_err_m, n_member)
    error_fnr = (HCHO_m/NO2_m)*np.sqrt((HCHO_err_m/HCHO_m)**2+(NO2_err_m/NO2_m)**2)
    error_fnr = np.abs(error_fnr)
    if np.isnan(error_fnr):
       error_fnr = 0.0
    s_fnr = np.random.normal(0, error_fnr, n_member)
    NO2_dist = NO2_m + s_no2
    HCHO_dist = HCHO_m + s_hcho
    FNR_dist = np.abs(HCHO_m/NO2_m) + s_fnr
    PO3_m = np.zeros((5, n_member))*np.nan
    for k in range(0, n_member):
        # estimate PO3
        threshold1 = 1.5
        threshold2 = 2.5
        threshold3 = 3.5
        if FNR_dist[k] < threshold1:
            coeff = COEFFs[0, :]
            coeff0 = COEFF0s[0]
        elif FNR_dist[k] > threshold3:
            coeff = COEFFs[1, :]
            coeff0 = COEFF0s[1]
        elif ((FNR_dist[k] >= threshold1) and (FNR_dist[k] < threshold2)):
            coeff = COEFFs[2, :]
            coeff0 = COEFF0s[2]
        elif ((FNR_dist[k] >= threshold2) and (FNR_dist[k] <= threshold3)):
            coeff = COEFFs[3, :]
            coeff0 = COEFF0s[3]
        else:
            coeff = np.zeros((4))*np.nan
            coeff0 = np.nan

        PO3_m[0, k] = J4_m*coeff[0]*1e3
        PO3_m[1, k] = J1_m*coeff[1]*1e6
        PO3_m[2, k] = HCHO_dist[k]*coeff[2]
        PO3_m[3, k] = NO2_dist[k]*coeff[3]
        PO3_m[4, k] = coeff0

    return np.mean(PO3_m, axis=1), np.std(PO3_m, axis=1)


def PO3est_empirical(no2_path, hcho_path, startdate, enddate, num_job=1):
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
    inputs = {
        "H2O": [], "FNR": [], "J1": [], "J4": [], "HCHO_ppbv": [], "NO2_ppbv": [],
        "SJ4": [], "SJ1": [], "SHCHO": [], "SNO2": [], "SH2O": [],
        "VCD_NO2": [], "VCD_FORM": [], "PBL_no2_factor": [], "PBL_form_factor": [],
        "PO3_err_sys": [], "PO3_err_rand": []
    }
    time_processed = []
    for single_date in _daterange(start_date, end_date):

        no2_files = sorted((glob.glob(no2_path + "/*_NO2_" + str(single_date.year) + f"{single_date.month:02}"
                                      + f"{single_date.day:02}" + "*.nc")))
        hcho_files = sorted((glob.glob(hcho_path + "/*_FORM_" + str(single_date.year) + f"{single_date.month:02}"
                                       + f"{single_date.day:02}" + "*.nc")))
        
        if (not no2_files) or (not hcho_files):
           print(f"files aren't available for {single_date}")
           continue
        # we make a list of inputs to append and average later for diags
        # Variables to accumulate daily data
        PBLH, VCD_NO2, VCD_FORM = [], [], []
        PBL_no2_factor, PBL_form_factor = [], []
        surface_albedo_no2, surface_albedo_hcho = [], []
        O3col, SZA, surface_alt = [], [], []
        VCD_NO2_err, VCD_HCHO_err, H2O = [], [], []
        # reading NO2 files daily
       # reading NO2 files daily
        for f in no2_files:
            print(f)
            PBLH.append(_read_nc(f, 'PBLH'))
            VCD_NO2.append(_read_nc(f, 'VCD'))
            PBL_no2_factor.append(_read_nc(f, 'gas_pbl_factor_no2'))
            #PL.append(_read_nc(f, 'pressure_mid'))
            #T.append(_read_nc(f, 'temperature_mid'))
            surface_albedo_no2.append(_read_nc(f, 'surface_albedo'))
            O3col.append(_read_nc(f, 'O3col'))
            SZA.append(_read_nc(f, 'SZA'))
            surface_alt.append(_read_nc(f, 'surface_alt'))
            latitude = _read_nc(f, 'latitude')
            longitude = _read_nc(f, 'longitude')
            VCD_NO2_err.append(_read_nc(f, 'VCD_err'))
            H2O.append(_read_nc(f, 'H2O'))
        # reading FORM files daily
        for f in hcho_files:
            VCD_FORM.append(_read_nc(f, 'VCD'))
            surface_albedo_hcho.append(_read_nc(f, 'surface_albedo'))
            PBL_form_factor.append(_read_nc(f, 'gas_pbl_factor_hcho'))
            VCD_HCHO_err.append(_read_nc(f, 'VCD_err'))

        # averaging to make daily coverage from L3 swaths
        PBLH = np.nanmean(np.array(PBLH), axis=0)
        VCD_NO2 = np.nanmean(np.array(VCD_NO2), axis=0)
        VCD_FORM = np.nanmean(np.array(VCD_FORM), axis=0)
        PBL_no2_factor = np.nanmean(np.array(PBL_no2_factor), axis=0)
        PBL_form_factor = np.nanmean(np.array(PBL_form_factor), axis=0)
        #PL = np.nanmean(np.array(PL), axis=0)
        #T = np.nanmean(np.array(T), axis=0)
        H2O = np.nanmean(np.array(H2O), axis=0)
        surface_albedo_no2 = np.nanmean(np.array(surface_albedo_no2), axis=0)
        surface_albedo_hcho = np.nanmean(np.array(surface_albedo_hcho), axis=0)
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

        # we can't handle negative values in this algorithm
        NO2_ppbv[NO2_ppbv<0] = 0.0
        HCHO_ppbv[HCHO_ppbv<0] = 0.0

        # taking care of random and sys errors
        NO2_ppbv_err_rand = VCD_NO2_err*PBL_no2_factor
        HCHO_ppbv_err_rand = VCD_HCHO_err*PBL_form_factor

        # the sys error consists of error in slope, error in offset, error in MINDS conversion factor
        NO2_ppbv_err_sys = np.sqrt(((0.01*VCD_NO2*PBL_no2_factor)**2 + (0.32*PBL_no2_factor)**2 +
                                   (VCD_NO2*0.09)**2))
        HCHO_ppbv_err_sys = np.sqrt(((0.01*VCD_FORM*PBL_form_factor)**2 + (0.90*PBL_form_factor)**2 +
                                    (VCD_FORM*0.08)**2))

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
        # for J1, we use surface HCHO albedo because it's located at a smaller wavelength
        J1 = interpn((SZAhybrid.flatten(), ALBhybrid.flatten(), O3Chybrid.flatten(), ALThybrid.flatten()),
                     J1, (SZA.flatten(), surface_albedo_hcho.flatten(),
                          O3col.flatten(), surface_alt.flatten()),
                     method="linear", bounds_error=False, fill_value=np.nan)
        J1 = np.reshape(J1, (np.shape(NO2_ppbv)[0], np.shape(NO2_ppbv)[1]))
        # load the lasso coeffs
        lasso_result = sio.loadmat('../data/lasso_piecewise_4group.mat')
        COEFF = lasso_result["COEFF"]
        COEFF0 = lasso_result["COEFF0"]
        COEFFs = np.zeros((4, 4))
        COEFF0s = np.zeros((4, 1))
        COEFFs[0, :] = np.array(COEFF[0, 0]).squeeze()
        COEFFs[1, :] = np.array(COEFF[0, 1]).squeeze()
        COEFFs[2, :] = np.array(COEFF[0, 2]).squeeze()
        COEFFs[3, :] = np.array(COEFF[0, 3]).squeeze()
        COEFF0s[0] = np.array(COEFF0[0, 0]).squeeze()
        COEFF0s[1] = np.array(COEFF0[0, 1]).squeeze()
        COEFF0s[2] = np.array(COEFF0[0, 2]).squeeze()
        COEFF0s[3] = np.array(COEFF0[0, 3]).squeeze()

        PO3 = np.zeros((np.shape(FNR)[0], np.shape(FNR)[1], 5))*np.nan
        PO3_err = np.zeros((np.shape(FNR)[0], np.shape(FNR)[1], 5))*np.nan
        # apply a monte-carlo way to approximate errors in PO3 estimates
        n_member = 10000
        output = Parallel(n_jobs=num_job)(delayed(loop_estimator)(
            J4[i, j], J1[i, j], HCHO_ppbv[i, j], NO2_ppbv[i, j], HCHO_ppbv_err_rand[i, j], NO2_ppbv_err_rand[i, j], COEFFs, COEFF0s, n_member) for i in range(0, np.shape(FNR)[0]) for j in range(0, np.shape(FNR)[1]))
        output = np.array(output)
        po3_dist = output[:,0,:].squeeze()
        po3_err_dist = output[:,1,:].squeeze()
        # integrating with PO3
        PO3[:, :, :] = po3_dist.reshape((np.shape(FNR)[0], np.shape(FNR)[1], 5))
        PO3_err[:, :, :] = po3_err_dist.reshape((np.shape(FNR)[0], np.shape(FNR)[1], 5))
        # append inputs and PO3_estimates daily
        PO3_estimates.append((np.sum(PO3, axis=2)))
        inputs["FNR"].append(np.abs(HCHO_ppbv/NO2_ppbv))
        inputs["H2O"].append(H2O)
        inputs["J1"].append(J1*1e6)
        inputs["J4"].append(J4*1e3)
        inputs["HCHO_ppbv"].append(HCHO_ppbv)
        inputs["NO2_ppbv"].append(NO2_ppbv)
        inputs["SJ4"].append(PO3[:, :, 0].squeeze())
        inputs["SJ1"].append(PO3[:, :, 1].squeeze())
        inputs["SH2O"].append(PO3[:, :, 2].squeeze()*0.0)
        inputs["SHCHO"].append(PO3[:, :, 2].squeeze())
        inputs["SNO2"].append(PO3[:, :, 3].squeeze())
        inputs["VCD_NO2"].append(VCD_NO2)
        inputs["VCD_FORM"].append(VCD_FORM)
        inputs["PBL_no2_factor"].append(PBL_no2_factor)
        inputs["PBL_form_factor"].append(PBL_form_factor)
        inputs["PO3_err_sys"].append(np.sqrt(np.sum(PO3_err**2*0.0, axis=2)))
        inputs["PO3_err_rand"].append(np.sqrt(np.sum(PO3_err**2, axis=2)))
        time_processed.append(datetime.datetime.combine(
            single_date, datetime.datetime.min.time()))

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
    PO3_err_sys = np.array(inputs["PO3_err_sys"])
    PO3_err_rand = np.array(inputs["PO3_err_rand"])

    output = param_output(latitude, longitude, time_processed, VCD_NO2, PBL_no2_factor, VCD_FORM, PBL_form_factor, PO3_estimates,
                          FNR, H2O, HCHO_ppbv, NO2_ppbv, J4, J1, HCHO_contrib, NO2_contrib, J4_contrib, J1_contrib, SH2O, PO3_err_rand, PO3_err_sys)
    return output
