import numpy as np
from math import pi
import scipy.io as io
import os
import matplotlib.pyplot as plt

def phasor_function(path_sample,sample_file,path_irfa,irf_file,real_data_en,threshold):
    
    threshold=int(threshold)
    
    DataSet = io.loadmat(os.path.join(path_sample,sample_file))
    if real_data_en:
        hist=DataSet.get('hist')
    else:
        hist=DataSet.get('decay')
            
    DataSet = io.loadmat(os.path.join(path_irfa,irf_file))
    irfa=DataSet.get('irf')
    
    # uo, io: uncalibrated phasor coordinate
    # ui, vi: calibrated phasor coordinate by eliminating the effect of IRF 
    # irf and hist should be aligned by irf_align beforehand
    
    sh = hist.shape
    si = irfa.shape
    
    cs = np.zeros((1,256))
    ss = np.zeros((1,256))
    
    if sh[1] != si[1]:
        print('No. of Hist and IRF time bins are unequal');
    else:
        T = sh[1]
    
    pixel_no = sh[0]
    
    
    t = np.arange(0.5,T,1)
    cs = np.cos(2*pi*t/T);
    ss = np.sin(2*pi*t/T);
    
    ui = np.zeros(sh[0])
    vi = np.zeros(sh[0])
    uo = np.zeros(sh[0])
    vo = np.zeros(sh[0])
    
    
    for i in range(pixel_no):
        y = hist[i,:]
        irf = irfa;
            
        intensity = y.sum();
        if (intensity < threshold):
            uo[i] = 0.0;
            vo[i] = 0.0;
            ui[i] = 0.0;
            vi[i] = 0.0;
        else:
            uo[i] = np.dot(y,cs)/intensity
            vo[i] = np.dot(y,ss)/intensity
            u_i = np.dot(irf,cs)/irf.sum()
            v_i = np.dot(irf,ss)/irf.sum()
            ui[i] = uo[i]*u_i + vo[i]*v_i
            vi[i] = -uo[i]*v_i + vo[i]*u_i

    return uo,vo,u_i,ui,vi

# Xedge=np.arange(0,1,0.002)
# Yedge=np.arange(0,1,0.002)

# plt.hist2d(ui, vi, bins=(Xedge, Yedge), range=None, density=False, weights=None, cmin=None, cmax=None, cmap=plt.cm.nipy_spectral)
# plt.show()
# x = np.arange(0,1.005,0.005)
# circle = np.sqrt(0.25 - (x - 0.5)**2)
# plt.plot(x,circle)
# plt.ylim(0,0.5)
# plt.xlim(0,1)
# plt.colorbar()
