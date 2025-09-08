import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import matplotlib.pyplot as plt
from scipy.io import loadmat,savemat
import matplotlib.cm as cm
from matplotlib import rcParams

p = pickle.load(open('data/normalization_weights.pkl', 'rb'))
dnn_model = keras.models.load_model('data/dnn_model_zscore.keras')

J1=40e-6 #30e-6 #70e-6
J4=12e-3 #10e-3 #13e-3 
H2O=0.4 #0.1 or 0.8

number_segment = 150

HCHO_ppbv = np.linspace(0,10,number_segment)
NO2_ppbv =  np.linspace(0,10,number_segment)


inputs_dnn = np.zeros((number_segment,number_segment,5))


for i in range(0,number_segment):
    for j in range(0,number_segment):
        inputs_dnn[:,:,0] = (J4-p[0]["jNO2"])/p[1]["jNO2"]
        inputs_dnn[:,:,1] = (J1-p[0]["jO1D"])/p[1]["jO1D"]
        inputs_dnn[:,:,2] = (H2O*1e18 - p[0]["H2O"])/p[1]["H2O"]
        inputs_dnn[i,j,3] = (NO2_ppbv[i] - p[0]["NO2"])/p[1]["NO2"]
        inputs_dnn[i,j,4] = (HCHO_ppbv[j] - p[0]["HCHO"])/p[1]["HCHO"]

inputs_dnn = inputs_dnn.reshape((number_segment*number_segment,5))
PO3 = np.array(dnn_model.predict(inputs_dnn, verbose=1))
PO3 = np.reshape(PO3,(number_segment,number_segment))
#fig, ax = plt.subplots()
#font = {'family' : 'normal',
#        'weight' : 'bold',
#        'size'   : 22}
#plt.rc('font', **font)

#cmap = plt.get_cmap('jet')

#im = ax.imshow(PO3.transpose(), interpolation='bilinear',
#               origin='lower', extent=[-3, 3, -3, 3],
#               vmax=30, vmin=-2.0, cmap=cmap)
#plt.show()

# Initialize plot objects

# Initialize plot objects
# Create a figure with 6 subplots (2 rows, 3 columns)
fig, axs = plt.subplots(2, 3, figsize=(17, 12))
# Generate a contour plot
#cp = ax.contour(NO2_ppbv, HCHO_ppbv, PO3)
#rcParams['figure.figsize'] = 5, 5 # sets plot size
#fig = plt.figure()
#ax = fig.add_subplot(111)

# Define levels in z-axis where we want lines to appear
levels = np.array(np.arange(0,np.max(PO3.flatten()),5))

# Generate a color mapping of the levels we've specified
import matplotlib.cm as cm # matplotlib's color map library
cpf = axs[0,0].contourf(NO2_ppbv, HCHO_ppbv, PO3.transpose(), len(levels), cmap=cm.Reds)

# Set all level lines to black
line_colors = ['black' for l in cpf.levels]

# Make plot and customize axes
cp = axs[0,0].contour(NO2_ppbv, HCHO_ppbv, PO3.transpose(), levels=levels, colors=line_colors,linewidths=4)
axs[0,0].clabel(cp, fontsize=25, colors=line_colors)
axs[0,0].set_xticks(np.arange(0,12,2))
axs[0,0].set_yticks(np.arange(0,12,2))
# Increase the tick label size
axs[0,0].tick_params(axis='both', which='major', labelsize=20)
_ = axs[0,0].set_ylabel('HCHO [ppbv]',fontsize=20)
axs[0,0].set_title("Norm",fontsize=30)
# Add a straight line
axs[0,0].axline((0, 0), slope=1.5, linewidth=5,color='blue')
axs[0,0].axline((0, 0), slope=2.5, linewidth=5,color='green')
axs[0,0].axline((0, 0), slope=3.5, linewidth=5,color='cyan')



#####

J1=70e-6 #30e-6 #70e-6
J4=14e-3 #10e-3 #13e-3 
H2O=0.4 #0.1 or 0.8

number_segment = 150

HCHO_ppbv = np.linspace(0,10,number_segment)
NO2_ppbv =  np.linspace(0,10,number_segment)


inputs_dnn = np.zeros((number_segment,number_segment,5))


for i in range(0,number_segment):
    for j in range(0,number_segment):
        inputs_dnn[:,:,0] = (J4-p[0]["jNO2"])/p[1]["jNO2"]
        inputs_dnn[:,:,1] = (J1-p[0]["jO1D"])/p[1]["jO1D"]
        inputs_dnn[:,:,2] = (H2O*1e18 - p[0]["H2O"])/p[1]["H2O"]
        inputs_dnn[i,j,3] = (NO2_ppbv[i] - p[0]["NO2"])/p[1]["NO2"]
        inputs_dnn[i,j,4] = (HCHO_ppbv[j] - p[0]["HCHO"])/p[1]["HCHO"]

inputs_dnn = inputs_dnn.reshape((number_segment*number_segment,5))
PO3 = np.array(dnn_model.predict(inputs_dnn, verbose=1))
PO3 = np.reshape(PO3,(number_segment,number_segment))
#fig, ax = plt.subplots()
#font = {'family' : 'normal',
#        'weight' : 'bold',
#        'size'   : 22}
#plt.rc('font', **font)

#cmap = plt.get_cmap('jet')

#im = ax.imshow(PO3.transpose(), interpolation='bilinear',
#               origin='lower', extent=[-3, 3, -3, 3],
#               vmax=30, vmin=-2.0, cmap=cmap)
#plt.show()

# Initialize plot objects

# Generate a contour plot
#cp = ax.contour(NO2_ppbv, HCHO_ppbv, PO3)
#rcParams['figure.figsize'] = 5, 5 # sets plot size
#fig = plt.figure()
#ax = fig.add_subplot(111)

# Define levels in z-axis where we want lines to appear
levels = np.array(np.arange(0,np.max(PO3.flatten()),5))

# Generate a color mapping of the levels we've specified
import matplotlib.cm as cm # matplotlib's color map library
cpf = axs[0,1].contourf(NO2_ppbv, HCHO_ppbv, PO3.transpose(), len(levels), cmap=cm.Reds)

# Set all level lines to black
line_colors = ['black' for l in cpf.levels]

# Make plot and customize axes
cp = axs[0,1].contour(NO2_ppbv, HCHO_ppbv, PO3.transpose(), levels=levels, colors=line_colors,linewidths=4)
axs[0,1].clabel(cp, fontsize=25, colors=line_colors)
axs[0,1].set_xticks(np.arange(0,12,2))
axs[0,1].set_yticks(np.arange(0,12,2))
axs[0,1].set_title("Bright",fontsize=30)
# Increase the tick label size
axs[0,1].tick_params(axis='both', which='major', labelsize=20)


##################

J1=30e-6 #30e-6 #70e-6
J4=7e-3 #10e-3 #13e-3 
H2O=0.4 #0.1 or 0.8

number_segment = 150

HCHO_ppbv = np.linspace(0,10,number_segment)
NO2_ppbv =  np.linspace(0,10,number_segment)


inputs_dnn = np.zeros((number_segment,number_segment,5))


for i in range(0,number_segment):
    for j in range(0,number_segment):
        inputs_dnn[:,:,0] = (J4-p[0]["jNO2"])/p[1]["jNO2"]
        inputs_dnn[:,:,1] = (J1-p[0]["jO1D"])/p[1]["jO1D"]
        inputs_dnn[:,:,2] = (H2O*1e18 - p[0]["H2O"])/p[1]["H2O"]
        inputs_dnn[i,j,3] = (NO2_ppbv[i] - p[0]["NO2"])/p[1]["NO2"]
        inputs_dnn[i,j,4] = (HCHO_ppbv[j] - p[0]["HCHO"])/p[1]["HCHO"]

inputs_dnn = inputs_dnn.reshape((number_segment*number_segment,5))
PO3 = np.array(dnn_model.predict(inputs_dnn, verbose=1))
PO3 = np.reshape(PO3,(number_segment,number_segment))
#fig, ax = plt.subplots()
#font = {'family' : 'normal',
#        'weight' : 'bold',
#        'size'   : 22}
#plt.rc('font', **font)

#cmap = plt.get_cmap('jet')

#im = ax.imshow(PO3.transpose(), interpolation='bilinear',
#               origin='lower', extent=[-3, 3, -3, 3],
#               vmax=30, vmin=-2.0, cmap=cmap)
#plt.show()

# Initialize plot objects

# Generate a contour plot
#cp = ax.contour(NO2_ppbv, HCHO_ppbv, PO3)
#rcParams['figure.figsize'] = 5, 5 # sets plot size
#fig = plt.figure()
#ax = fig.add_subplot(111)

# Define levels in z-axis where we want lines to appear
levels = np.array(np.arange(0,np.max(PO3.flatten()),5))

# Generate a color mapping of the levels we've specified
import matplotlib.cm as cm # matplotlib's color map library
cpf = axs[0,2].contourf(NO2_ppbv, HCHO_ppbv, PO3.transpose(), len(levels), cmap=cm.Reds)

# Set all level lines to black
line_colors = ['black' for l in cpf.levels]

# Make plot and customize axes
cp = axs[0,2].contour(NO2_ppbv, HCHO_ppbv, PO3.transpose(), levels=levels, colors=line_colors,linewidths=4)
axs[0,2].clabel(cp, fontsize=25, colors=line_colors)
axs[0,2].set_xticks(np.arange(0,12,2))
axs[0,2].set_yticks(np.arange(0,12,2))
axs[0,2].set_title("Dim",fontsize=30)
# Increase the tick label size
axs[0,2].tick_params(axis='both', which='major', labelsize=20)

######################3

J1=40e-6 #30e-6 #70e-6
J4=12e-3 #10e-3 #13e-3 
H2O=0.1 #0.1 or 0.8

number_segment = 150

HCHO_ppbv = np.linspace(0,10,number_segment)
NO2_ppbv =  np.linspace(0,10,number_segment)


inputs_dnn = np.zeros((number_segment,number_segment,5))


for i in range(0,number_segment):
    for j in range(0,number_segment):
        inputs_dnn[:,:,0] = (J4-p[0]["jNO2"])/p[1]["jNO2"]
        inputs_dnn[:,:,1] = (J1-p[0]["jO1D"])/p[1]["jO1D"]
        inputs_dnn[:,:,2] = (H2O*1e18 - p[0]["H2O"])/p[1]["H2O"]
        inputs_dnn[i,j,3] = (NO2_ppbv[i] - p[0]["NO2"])/p[1]["NO2"]
        inputs_dnn[i,j,4] = (HCHO_ppbv[j] - p[0]["HCHO"])/p[1]["HCHO"]

inputs_dnn = inputs_dnn.reshape((number_segment*number_segment,5))
PO3 = np.array(dnn_model.predict(inputs_dnn, verbose=1))
PO3 = np.reshape(PO3,(number_segment,number_segment))
#fig, ax = plt.subplots()
#font = {'family' : 'normal',
#        'weight' : 'bold',
#        'size'   : 22}
#plt.rc('font', **font)

#cmap = plt.get_cmap('jet')

#im = ax.imshow(PO3.transpose(), interpolation='bilinear',
#               origin='lower', extent=[-3, 3, -3, 3],
#               vmax=30, vmin=-2.0, cmap=cmap)
#plt.show()

# Initialize plot objects
# Generate a contour plot
#cp = ax.contour(NO2_ppbv, HCHO_ppbv, PO3)
#rcParams['figure.figsize'] = 5, 5 # sets plot size
#fig = plt.figure()
#ax = fig.add_subplot(111)

# Define levels in z-axis where we want lines to appear
levels = np.array(np.arange(0,np.max(PO3.flatten()),5))

# Generate a color mapping of the levels we've specified
import matplotlib.cm as cm # matplotlib's color map library
cpf = axs[1,0].contourf(NO2_ppbv, HCHO_ppbv, PO3.transpose(), len(levels), cmap=cm.Reds)

# Set all level lines to black
line_colors = ['black' for l in cpf.levels]

# Make plot and customize axes
cp = axs[1,0].contour(NO2_ppbv, HCHO_ppbv, PO3.transpose(), levels=levels, colors=line_colors,linewidths=4)
axs[1,0].clabel(cp, fontsize=25, colors=line_colors)
axs[1,0].set_xticks(np.arange(0,12,2))
axs[1,0].set_yticks(np.arange(0,12,2))
# Increase the tick label size
axs[1,0].tick_params(axis='both', which='major', labelsize=20)
axs[1,0].set_xlabel('NO2 [ppbv]',fontsize=20)
_ = axs[1,0].set_ylabel('HCHO [ppbv]',fontsize=20)
axs[1,0].set_title("Dry",fontsize=30)
##############


J1=40e-6 #30e-6 #70e-6
J4=12e-3 #10e-3 #13e-3 
H2O=0.8 #0.1 or 0.8

number_segment = 150

HCHO_ppbv = np.linspace(0,10,number_segment)
NO2_ppbv =  np.linspace(0,10,number_segment)


inputs_dnn = np.zeros((number_segment,number_segment,5))


for i in range(0,number_segment):
    for j in range(0,number_segment):
        inputs_dnn[:,:,0] = (J4-p[0]["jNO2"])/p[1]["jNO2"]
        inputs_dnn[:,:,1] = (J1-p[0]["jO1D"])/p[1]["jO1D"]
        inputs_dnn[:,:,2] = (H2O*1e18 - p[0]["H2O"])/p[1]["H2O"]
        inputs_dnn[i,j,3] = (NO2_ppbv[i] - p[0]["NO2"])/p[1]["NO2"]
        inputs_dnn[i,j,4] = (HCHO_ppbv[j] - p[0]["HCHO"])/p[1]["HCHO"]

inputs_dnn = inputs_dnn.reshape((number_segment*number_segment,5))
PO3 = np.array(dnn_model.predict(inputs_dnn, verbose=1))
PO3 = np.reshape(PO3,(number_segment,number_segment))
#fig, ax = plt.subplots()
#font = {'family' : 'normal',
#        'weight' : 'bold',
#        'size'   : 22}
#plt.rc('font', **font)

#cmap = plt.get_cmap('jet')

#im = ax.imshow(PO3.transpose(), interpolation='bilinear',
#               origin='lower', extent=[-3, 3, -3, 3],
#               vmax=30, vmin=-2.0, cmap=cmap)
#plt.show()

# Initialize plot objects
# Generate a contour plot
#cp = ax.contour(NO2_ppbv, HCHO_ppbv, PO3)
#rcParams['figure.figsize'] = 5, 5 # sets plot size
#fig = plt.figure()
#ax = fig.add_subplot(111)

# Define levels in z-axis where we want lines to appear
levels = np.array(np.arange(0,np.max(PO3.flatten()),5))

# Generate a color mapping of the levels we've specified
import matplotlib.cm as cm # matplotlib's color map library
cpf = axs[1,1].contourf(NO2_ppbv, HCHO_ppbv, PO3.transpose(), len(levels), cmap=cm.Reds)

# Set all level lines to black
line_colors = ['black' for l in cpf.levels]

# Make plot and customize axes
cp = axs[1,1].contour(NO2_ppbv, HCHO_ppbv, PO3.transpose(), levels=levels, colors=line_colors,linewidths=4)
axs[1,1].clabel(cp, fontsize=25, colors=line_colors)
axs[1,1].set_xticks(np.arange(0,12,2))
axs[1,1].set_yticks(np.arange(0,12,2))
# Increase the tick label size
axs[1,1].tick_params(axis='both', which='major', labelsize=20)
axs[1,1].set_xlabel('NO2 [ppbv]',fontsize=20)
axs[1,1].set_title("Humid",fontsize=30)

##################33

J1=70e-6 #30e-6 #70e-6
J4=14e-3 #10e-3 #13e-3 
H2O=0.8 #0.1 or 0.8

number_segment = 150

HCHO_ppbv = np.linspace(0,10,number_segment)
NO2_ppbv =  np.linspace(0,10,number_segment)


inputs_dnn = np.zeros((number_segment,number_segment,5))


for i in range(0,number_segment):
    for j in range(0,number_segment):
        inputs_dnn[:,:,0] = (J4-p[0]["jNO2"])/p[1]["jNO2"]
        inputs_dnn[:,:,1] = (J1-p[0]["jO1D"])/p[1]["jO1D"]
        inputs_dnn[:,:,2] = (H2O*1e18 - p[0]["H2O"])/p[1]["H2O"]
        inputs_dnn[i,j,3] = (NO2_ppbv[i] - p[0]["NO2"])/p[1]["NO2"]
        inputs_dnn[i,j,4] = (HCHO_ppbv[j] - p[0]["HCHO"])/p[1]["HCHO"]

inputs_dnn = inputs_dnn.reshape((number_segment*number_segment,5))
PO3 = np.array(dnn_model.predict(inputs_dnn, verbose=1))
PO3 = np.reshape(PO3,(number_segment,number_segment))
#fig, ax = plt.subplots()
#font = {'family' : 'normal',
#        'weight' : 'bold',
#        'size'   : 22}
#plt.rc('font', **font)

#cmap = plt.get_cmap('jet')

#im = ax.imshow(PO3.transpose(), interpolation='bilinear',
#               origin='lower', extent=[-3, 3, -3, 3],
#               vmax=30, vmin=-2.0, cmap=cmap)
#plt.show()

# Initialize plot objects
# Generate a contour plot
#cp = ax.contour(NO2_ppbv, HCHO_ppbv, PO3)
#rcParams['figure.figsize'] = 5, 5 # sets plot size
#fig = plt.figure()
#ax = fig.add_subplot(111)

# Define levels in z-axis where we want lines to appear
levels = np.array(np.arange(0,np.max(PO3.flatten()),5))

# Generate a color mapping of the levels we've specified
import matplotlib.cm as cm # matplotlib's color map library
cpf = axs[1,2].contourf(NO2_ppbv, HCHO_ppbv, PO3.transpose(), len(levels), cmap=cm.Reds)

# Set all level lines to black
line_colors = ['black' for l in cpf.levels]

# Make plot and customize axes
cp = axs[1,2].contour(NO2_ppbv, HCHO_ppbv, PO3.transpose(), levels=levels, colors=line_colors,linewidths=4)
axs[1,2].clabel(cp, fontsize=25, colors=line_colors)
axs[1,2].set_xticks(np.arange(0,12,2))
axs[1,2].set_yticks(np.arange(0,12,2))
# Increase the tick label size
axs[1,2].tick_params(axis='both', which='major', labelsize=20)
axs[1,2].set_xlabel('NO2 [ppbv]',fontsize=20)
axs[1,2].set_title("Humid and Bright",fontsize=30)


plt.savefig('isopleths.png', dpi=300)
