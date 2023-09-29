import scipy.io as sio
import numpy as np
import glob
import warnings
from netCDF4 import Dataset
from scipy.interpolate import interpn
import datetime

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

    start_date = datetime.date(int(startdate[0:4]), int(
        startdate[5:7]), int(startdate[8:10]))
    end_date = datetime.date(int(enddate[0:4]), int(
        enddate[5:7]), int(enddate[8:10]))
    
    PO3_estimates = []
    input1 = []
    input2 = []
    input3 = []
    input4 = []
    input5 = []    
    for single_date in _daterange(start_date, end_date):

        no2_files = sorted((glob.glob(no2_path + "/*_NO2_" + str(single_date.year) + f"{single_date.month:02}"
                                       + f"{single_date.day:02}" + "*.nc")))
        hcho_files = sorted((glob.glob(hcho_path + "/*_FORM_" + str(single_date.year) + f"{single_date.month:02}"
                                       + f"{single_date.day:02}" + "*.nc")))
        
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
        for f in no2_files:
            print(f)
            PBLH.append(_read_nc(f, 'PBLH'))
            VCD_NO2.append(_read_nc(f, 'VCD'))
            PBL_no2_factor.append(_read_nc(f, 'gas_pbl_factor_no2'))
            PL.append(_read_nc(f, 'pressure_mid'))
            T.append(_read_nc(f, 'tempeature_mid'))
            surface_albedo_no2.append(_read_nc(f, 'surface_albedo'))
            O3col.append(_read_nc(f, 'O3col'))
            SZA.append(_read_nc(f, 'SZA'))
            surface_alt.append(_read_nc(f, 'surface_alt'))

        for f in hcho_files:
            VCD_FORM.append(_read_nc(f, 'VCD'))
            PBL_form_factor.append(_read_nc(f, 'gas_pbl_factor_hcho'))

        PBLH = np.nanmean(np.array(PBLH),axis=0)
        VCD_NO2 = np.nanmean(np.array(VCD_NO2),axis=0)
        VCD_FORM = np.nanmean(np.array(VCD_FORM),axis=0)
        PBL_no2_factor = np.nanmean(np.array(PBL_no2_factor),axis=0)
        PBL_form_factor = np.nanmean(np.array(PBL_form_factor),axis=0)
        PL = np.nanmean(np.array(PL),axis=0)
        T = np.nanmean(np.array(T),axis=0)
        surface_albedo_no2 = np.nanmean(np.array(surface_albedo_no2),axis=0)
        O3col = np.nanmean(np.array(O3col),axis=0)
        SZA = np.nanmean(np.array(SZA),axis=0)
        surface_alt = np.nanmean(np.array(surface_alt),axis=0)
        # extract the features
        # potential temp, HCHO_ppbv, NO2_ppbv, jNO2, FNR
        mask_PBL = PL >= PBLH
        mask_PBL = np.multiply(mask_PBL, 1.0).squeeze()
        mask_PBL[mask_PBL != 1.0] = np.nan
        potential_temp = np.nansum(T*((1000/PL)**(0.286))*mask_PBL, axis=0)
        NO2_ppbv = VCD_NO2*PBL_no2_factor
        HCHO_ppbv = VCD_FORM*PBL_form_factor
        FNR = np.log(HCHO_ppbv/NO2_ppbv)
        Jvalue = sio.loadmat('HybridJtables.mat')
        SZAhybrid = Jvalue["SZAhybrid"][:, 0, 0, 0]
        ALBhybrid = Jvalue["ALBhybrid"][0, :, 0, 0]
        O3Chybrid = Jvalue["O3Chybrid"][0, 0, :, 0]
        ALThybrid = Jvalue["ALThybrid"][0, 0, 0, :]
        J = Jvalue["Jhybrid"]
        J4 = (J["J4"])
        J4 = np.array(J4[0, 0])
        J4 = interpn((SZAhybrid.flatten(), ALBhybrid.flatten(), O3Chybrid.flatten(), ALThybrid.flatten()),
                     J4, (SZA.flatten(), surface_albedo_no2.flatten(), O3col.flatten(), surface_alt.flatten()), 
                     method="linear",bounds_error=False,fill_value=np.nan)
        J4 = np.reshape(J4, (np.shape(FNR)[0], np.shape(FNR)[1]))

        # load the lasso coeffs
        lasso_result = sio.loadmat('lasso_piecewise.mat')
        COEFF = lasso_result["COEFF"]
        COEFF0 = lasso_result["COEFF0"]
        COEFF1 = np.array(COEFF[0, 0])
        COEFF2 = np.array(COEFF[0, 1])
        COEFF01 = np.array(COEFF0[0, 0])
        COEFF02 = np.array(COEFF0[0, 1])
        PO3 = np.zeros_like(FNR)*np.nan
        for i in range(0, np.shape(FNR)[0]):
            for j in range(0, np.shape(FNR)[1]):
                if FNR[i, j] < 4:
                    coeff = COEFF1
                    coeff0 = COEFF01
                else:
                    coeff = COEFF2
                    coeff0 = COEFF02
                PO3[i, j] = np.log(FNR[i, j])*coeff[0]
                PO3[i, j] = PO3[i, j]+potential_temp[i, j]*coeff[1]
                PO3[i, j] = PO3[i, j]+J4[i, j]*coeff[2]*1e3
                PO3[i, j] = PO3[i, j]+HCHO_ppbv[i, j]*coeff[3]
                PO3[i, j] = PO3[i, j]+NO2_ppbv[i, j]*coeff[4]
                PO3[i, j] = PO3[i, j]+coeff0
        
        PO3_estimates.append(PO3)
        input1.append(np.log(FNR))
        input2.append(potential_temp)
        input3.append(J4*1e3)
        input4.append(HCHO_ppbv)
        input5.append(NO2_ppbv)
    
    mdic = {"PO3":np.array(PO3_estimates),"input1":np.array(input1),"input2":np.array(input2),
            "input3":np.array(input3),"input4":np.array(input4),"input5":np.array(input5)}
    sio.savemat("PO3.mat", mdic)
