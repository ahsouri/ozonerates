import numpy as np
import scipy.io as sio

lasso_result = sio.loadmat('ozonerates/lasso_piecewise.mat')
COEFF = lasso_result["COEFF"]
COEFF1= np.array(COEFF[0,0])
print(np.shape(COEFF1))