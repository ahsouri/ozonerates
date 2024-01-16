import scipy.io as sio
import numpy as np
import glob
import warnings
from netCDF4 import Dataset
from scipy.interpolate import interpn
import datetime
from scipy.io import savemat
from ozonerates.config import param_output

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


def PO3est_empirical(no2_path, hcho_path, startdate, enddate):
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
    inputs["FNR"]=[]
    inputs["J1"]=[]
    inputs["J4"]=[]
    inputs["HCHO_ppbv"]=[]
    inputs["NO2_ppbv"]=[]
    inputs["PO3_J4"]=[]
    inputs["PO3_J1"]=[]
    inputs["PO3_HCHO"]=[]
    inputs["PO3_NO2"]=[]
    inputs["VCD_NO2"]=[]
    inputs["VCD_FORM"]=[]
    inputs["PBL_no2_factor"]=[]
    inputs["PBL_form_factor"]=[]
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

        # reading FORM files daily
        for f in hcho_files:
            VCD_FORM.append(_read_nc(f, 'VCD'))
            PBL_form_factor.append(_read_nc(f, 'gas_pbl_factor_hcho'))

        # averaging to make daily coverage from L3 swaths
        PBLH = np.nanmean(np.array(PBLH), axis=0)
        VCD_NO2 = np.nanmean(np.array(VCD_NO2), axis=0)
        VCD_FORM = np.nanmean(np.array(VCD_FORM), axis=0)
        PBL_no2_factor = np.nanmean(np.array(PBL_no2_factor), axis=0)
        PBL_form_factor = np.nanmean(np.array(PBL_form_factor), axis=0)
        PL = np.nanmean(np.array(PL), axis=0)
        T = np.nanmean(np.array(T), axis=0)
        surface_albedo_no2 = np.nanmean(np.array(surface_albedo_no2), axis=0)
        O3col = np.nanmean(np.array(O3col), axis=0)
        SZA = np.nanmean(np.array(SZA), axis=0)
        surface_alt = np.nanmean(np.array(surface_alt), axis=0)
        surface_alt[surface_alt<=0] = 0.0 #oceanic areas sometimes are negative
        # extract the features: potential temp, HCHO_ppbv, NO2_ppbv, jNO2, FNR
        mask_PBL = PL >= PBLH
        mask_PBL = np.multiply(mask_PBL, 1.0).squeeze()
        mask_PBL[mask_PBL != 1.0] = np.nan
        #potential_temp = np.nanmean(T*((1000/PL)**(0.286))*mask_PBL, axis=0)
        NO2_ppbv = VCD_NO2*PBL_no2_factor
        HCHO_ppbv = VCD_FORM*PBL_form_factor
        FNR = (HCHO_ppbv/NO2_ppbv)

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

        J1 = (J["J1"])
        J1 = np.array(J1[0, 0])
        # linear interpolation (extrapolation is not allowed = NaN)
        J4 = interpn((SZAhybrid.flatten(), ALBhybrid.flatten(), O3Chybrid.flatten(), ALThybrid.flatten()),
                     J4, (SZA.flatten(), surface_albedo_no2.flatten(),
                          O3col.flatten(), surface_alt.flatten()),
                     method="linear", bounds_error=False, fill_value=np.nan)
        J4 = np.reshape(J4, (np.shape(FNR)[0], np.shape(FNR)[1]))
        J1 = interpn((SZAhybrid.flatten(), ALBhybrid.flatten(), O3Chybrid.flatten(), ALThybrid.flatten()),
                     J1, (SZA.flatten(), surface_albedo_no2.flatten(),
                          O3col.flatten(), surface_alt.flatten()),
                     method="linear", bounds_error=False, fill_value=np.nan)
        J1 = np.reshape(J1, (np.shape(FNR)[0], np.shape(FNR)[1]))
        # load the lasso coeffs
        lasso_result = sio.loadmat('../data/lasso_piecewise_3group.mat')
        COEFF = lasso_result["COEFF"]
        COEFF0 = lasso_result["COEFF0"]
        COEFF1 = np.array(COEFF[0, 0])
        COEFF2 = np.array(COEFF[0, 1])
        COEFF3 = np.array(COEFF[0, 2])
        COEFF01 = np.array(COEFF0[0, 0])
        COEFF02 = np.array(COEFF0[0, 1])
        COEFF03 = np.array(COEFF0[0, 2])
        # estimate PO3
        threshold1 = 2
        threshold2 = 4
        PO3 = np.zeros((np.shape(FNR)[0], np.shape(FNR)[1], 5))*np.nan
        for i in range(0, np.shape(FNR)[0]):
            for j in range(0, np.shape(FNR)[1]):
                if FNR[i, j] < threshold1:
                    coeff = COEFF1
                    coeff0 = COEFF01
                elif FNR[i, j] > threshold2:
                    coeff = COEFF2
                    coeff0 = COEFF02
                elif ((FNR[i, j] >= threshold1) and (FNR[i, j] <= threshold2)):
                    coeff = COEFF3
                    coeff0 = COEFF03
                else:
                    continue

                #PO3[i, j] = PO3[i, j]+(FNR[i, j])*coeff[0]
                #PO3[i, j] = PO3[i, j]+potential_temp[i, j]*coeff[1]
                PO3[i, j, 0] = J4[i, j]*coeff[0]*1e3
                PO3[i, j, 1] = J1[i, j]*coeff[1]*1e6
                PO3[i, j, 2] = HCHO_ppbv[i, j]*coeff[2]
                PO3[i, j, 3] = NO2_ppbv[i, j]*coeff[3]
                PO3[i, j, 4] = coeff0

        # append inputs and PO3_estimates daily
        PO3_estimates.append(np.sum(PO3, axis=2))
        inputs["FNR"].append(FNR)
        inputs["J1"].append(J1*1e6)
        inputs["J4"].append(J4*1e3)
        inputs["HCHO_ppbv"].append(HCHO_ppbv)
        inputs["NO2_ppbv"].append(NO2_ppbv)
        inputs["PO3_J4"].append(PO3[:, :, 0].squeeze())
        inputs["PO3_J1"].append(PO3[:, :, 1].squeeze())
        inputs["PO3_HCHO"].append(PO3[:, :, 2].squeeze())
        inputs["PO3_NO2"].append(PO3[:, :, 3].squeeze())
        inputs["VCD_NO2"].append(VCD_NO2)
        inputs["VCD_FORM"].append(VCD_FORM)
        inputs["PBL_no2_factor"].append(PBL_no2_factor)
        inputs["PBL_form_factor"].append(PBL_form_factor)

    FNR = np.array(inputs["FNR"])
    J1 = np.array(inputs["J1"])
    J4 = np.array(inputs["J4"])
    HCHO_ppbv = np.array(inputs["HCHO_ppbv"])
    NO2_ppbv = np.array(inputs["NO2_ppbv"])
    J1_contrib = np.array(inputs["PO3_J1"])
    J4_contrib = np.array(inputs["PO3_J4"])
    NO2_contrib = np.array(inputs["PO3_NO2"])
    HCHO_contrib = np.array(inputs["PO3_HCHO"])
    VCD_NO2 = np.array(inputs["VCD_NO2"])
    VCD_FORM = np.array(inputs["VCD_FORM"])
    PBL_no2_factor = np.array(inputs["PBL_no2_factor"])
    PBL_form_factor = np.array(inputs["PBL_form_factor"])
    PO3_estimates = np.array(PO3_estimates)

    output = param_output(latitude, longitude, VCD_NO2, PBL_no2_factor, VCD_FORM, PBL_form_factor, PO3_estimates,
                          FNR, HCHO_ppbv, NO2_ppbv, J4, J1, HCHO_contrib, NO2_contrib, J4_contrib, J1_contrib)
    return output
