# -*- coding: utf-8 -*-
#ZZY 19/July/2022

import os
import torch
import numpy as np
import scipy.io as io
from CNN_Two_o import S_Tau_Net_Two_o
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.io import savemat
import time
from torchsummary import summary
from torchstat import stat
from thop import profile
from pynvml import *
#%%
nvmlInit()
print ("Driver Version:", nvmlSystemGetDriverVersion())
deviceCount = nvmlDeviceGetCount()
for i in range(deviceCount):
    handle = nvmlDeviceGetHandleByIndex(i)
    print ("Device", i, ":", nvmlDeviceGetName(handle))

power_start=nvmlDeviceGetPowerUsage(handle)

print ("Start Power:", power_start)
#%%
use_quan=0
use_cpu=0

np.set_printoptions(suppress=True)
torch.set_printoptions(precision=5)

#mouse cells
path=r'C:\Users\Zhenya\Desktop\ELM2021\FLIM_code_10_July\Sample\cell data'
sample_file='hist_6h1_lf82.mat'

DataSet = io.loadmat(os.path.join(path,sample_file))
hist=DataSet.get('hist65536')  
hist_nor=DataSet.get('hist65536_nor')

hist=hist.astype(np.float32)
hist=hist.reshape(hist.shape[0],1,hist.shape[1])
hist = Variable(torch.from_numpy(hist))

hist_nor=hist_nor.astype(np.float32)
hist_nor=hist_nor.reshape(hist_nor.shape[0],1,hist_nor.shape[1])
hist_nor = Variable(torch.from_numpy(hist_nor))

#use gpu
PATH=r'./CNN_val_loss_0.060866.pth'
model = S_Tau_Net_Two_o()

#use cpu
if use_cpu==1:
    model.to(device="cpu")    
    model.eval()
    hist=torch.unsqueeze(hist,0)
    checkpoint=torch.load(PATH,map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint,strict=False)
    
    tau_amp=np.zeros(65536)
    tau_inten=np.zeros(65536)
    threshold=100
    processed_pixels=0
    
    hist_nothre=hist_nor.unsqueeze(2)
    hist_nothre=hist_nothre.unsqueeze(2)
    
    print(hist_nothre.device)
    
    start = time.time()
    with torch.no_grad():
        for i in range(65536):
            hist_thre=torch.squeeze(torch.squeeze(hist))
            if torch.sum(hist_thre[i,:],dim=0)<threshold:
                yhat=0
            else:
                yhat = model(hist_nothre[i,:,:,:])
                tau_amp[i]=yhat[0]
                tau_inten[i]=yhat[1]
                processed_pixels=+1
            if (i%10000==0):
                print('{} pixel processed'.format(i))
    
    tau_amp=tau_amp.reshape(256,256)
    tau_inten=tau_inten.reshape(256,256)
    
    end = time.time()
    time_consumption=end - start
    HPS=float(processed_pixels/time_consumption)
    print('------Inference Done------')
    print(f"Runtime of the program is {time_consumption}")
    print(f"Histogram per second is {HPS}")
else:
    device = torch.device("cuda")
    model.eval()
    model=model.cuda()
    hist=torch.unsqueeze(hist,0)
    checkpoint=torch.load(PATH,map_location="cuda:0")
    model.load_state_dict(checkpoint,strict=False)
    
    tau_amp=np.zeros(65536)
    tau_inten=np.zeros(65536)
    threshold=1000
    processed_pixels=0
    
    hist_nothre=hist_nor.unsqueeze(2)
    hist_nothre=hist_nothre.cuda()
    print(hist_nothre.device)
     
    start = time.time()
    with torch.no_grad():
        for i in range(65536):
            hist_thre=torch.squeeze(torch.squeeze(hist))
            if torch.sum(hist_thre[i,:],dim=0)<threshold:
                yhat=0
            else:
                yhat = model(hist_nothre[i,:,:,:])
                tau_amp[i]=yhat[0]
                tau_inten[i]=yhat[1]
                processed_pixels=+1
            if (i%10000==0):
                print('{} pixel processed'.format(i))
    
    tau_amp=tau_amp.reshape(256,256)
    tau_inten=tau_inten.reshape(256,256)
    
    end = time.time()
    time_consumption=end - start
    HPS=float(processed_pixels/time_consumption)
    print('------Inference Done------')
    print(f"Runtime of the program is {time_consumption}")
    print(f"Histogram per second is {HPS}")
    
tau_amp=np.transpose(tau_amp,[1,0])
tau_inten=np.transpose(tau_inten,[1,0])

power_end=nvmlDeviceGetPowerUsage(handle)
print ("End Power:", power_end)

power_consumpion=power_end-power_start
print ("power_consumpion:", power_consumpion)

device_util_rate=nvmlDeviceGetUtilizationRates(handle)
print ("device_util_rate:", device_util_rate)

nvmlShutdown()

#%%For samples
plt.imshow(tau_amp, cmap='nipy_spectral')#nipy_spectral, 
cb=plt.colorbar()
cb.ax.tick_params(labelsize=15)
plt.clim(0, 3.5)
plt.axis('off')
plt.show()
#%%
plt.imshow(tau_inten, cmap='nipy_spectral')#nipy_spectral, gist_ncar,gist_stern
cb=plt.colorbar()
cb.ax.tick_params(labelsize=15)
plt.clim(0, 3.5)
plt.axis('off')
plt.show()
#%%plot YG GT
path=r'C:\Users\Zhenya\Desktop\1DCNN_no_padding\S_TauNet_two_output\mat_result'
sample_file='mouse_cell_phasor_proj.mat'
DataSet = io.loadmat(os.path.join(path,sample_file))
hist=DataSet.get('pp_image')  
plt.imshow(hist, cmap='gist_ncar')#nipy_spectral, gist_ncar,gist_stern
# cb=plt.colorbar()
# cb.ax.tick_params(labelsize=15)
plt.clim(0, 1)
plt.axis('off')
plt.show()