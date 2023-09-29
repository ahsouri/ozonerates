import scipy.io as sio
import numpy as np
import glob
import warnings
from netCDF4 import Dataset
from scipy.interpolate import interpn
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _read_nc(filename, var):
    # reading nc files without a group
    nc_f = filename
    nc_fid = Dataset(nc_f, 'r')
    out = np.array(nc_fid.variables[var])
    nc_fid.close()
    return np.squeeze(out)


def PO3est_empirical(no2_path, hcho_path):
    no2_files = sorted((glob.glob(no2_path + "/*_NO2_*.nc")))
    PO3_estimates = []
    for f in no2_files:
        print(f)
        PBLH = _read_nc(f, 'PBLH')
        VCD_NO2 = _read_nc(f, 'VCD')
        no2_pbl_factor = _read_nc(f, 'gas_pbl_factor_no2')
        PL = _read_nc(f, 'pressure_mid')
        T = _read_nc(f, 'tempeature_mid')
        surface_albedo_no2 = _read_nc(f, 'surface_albedo')
        O3col = _read_nc(f, 'O3col')
        SZA = _read_nc(f, 'SZA')
        surface_alt = _read_nc(f, 'surface_alt')
        date_fname = f.split("NO2")
        form_file = ((glob.glob(hcho_path + "/*_FORM" + date_fname[1])))
        if not form_file: continue
        VCD_FORM = _read_nc(form_file[0], 'VCD')
        hcho_pbl_factor = _read_nc(f, 'gas_pbl_factor_hcho')
        # extract the features
        # potential temp, HCHO_ppbv, NO2_ppbv, jNO2, FNR
        mask_PBL = PL >= PBLH
        mask_PBL = np.multiply(mask_PBL, 1.0).squeeze()
        mask_PBL[mask_PBL != 1.0] = np.nan
        potential_temp = np.nansum(T*((1000/PL)**(0.286))*mask_PBL, axis=0)
        NO2_ppbv = VCD_NO2*no2_pbl_factor
        HCHO_ppbv = VCD_FORM*hcho_pbl_factor
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
        J4 = np.reshape(J4, (np.shape(FNR)[0], np.shape(FNR)[1]))*1e3

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
                PO3[i, j] = PO3[i, j]+J4[i, j]*coeff[2]
                PO3[i, j] = PO3[i, j]+HCHO_ppbv[i, j]*coeff[3]
                PO3[i, j] = PO3[i, j]+NO2_ppbv[i, j]*coeff[4]
                PO3[i, j] = PO3[i, j]+coeff0
        PO3_estimates.append(PO3)
        if len(PO3_estimates)>30:
           break
    PO3_estimates = np.array(PO3_estimates)
    print(np.shape(PO3_estimates))
    mdic = {"PO3":PO3_estimates}
    sio.savemat("PO3.mat", mdic)
