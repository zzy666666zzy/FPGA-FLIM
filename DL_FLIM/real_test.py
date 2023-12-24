# -*- coding: utf-8 -*-
#ZZY 02/June/2021

import os
import torch
import numpy as np
import scipy.io as io
from S_TauNet_AVE_model import S_TauNet_AVE
from Q_S_TauNet_AVE_model import Quan_S_TauNet_AVE
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.io import savemat
import time
#from torchsummary import summary
#from torchstat import stat
#from thop import profile
#from pynvml import *
#%%
#nvmlInit()
#print ("Driver Version:", nvmlSystemGetDriverVersion())
#deviceCount = nvmlDeviceGetCount()
#for i in range(deviceCount):
#    handle = nvmlDeviceGetHandleByIndex(i)
#    print ("Device", i, ":", nvmlDeviceGetName(handle))
#
#power_start=nvmlDeviceGetPowerUsage(handle)
#
#print ("Start Power:", power_start)
#%%
use_quan=0
use_cpu=0

np.set_printoptions(suppress=True)
torch.set_printoptions(precision=5)


#---for complete data test-----
#with gold samples
# path=r'C:\Users\Zhenya\Desktop\S_Tau_Net_conference'
# sample_file='PC_100_cycle.mat'

# path=r'C:\Users\Zhenya\Desktop\ELM2021\FLIM_code_10_July\beads samples'
# sample_file='beads_100c_p6.mat'

#mouse cells
#path=r'C:\Users\Zhenya\Desktop\ELM2021\FLIM_code_10_July\Sample\cell data'
#sample_file='hist_6h1_lf82.mat'

#plant
path=r'C:\Users\Administrator\Desktop\FLIM_code_10_July\Sample'
sample_file='hist_plant_cell_1cyle.mat'

# path=r'C:\Users\Zhenya\Desktop\ELM2021\FLIM_code_10_July\Sample'
# sample_file='pc3_probe1.mat'

# path=r'C:\Users\Zhenya\Desktop\ELM2021\FLIM_code_10_July\Sample'
# sample_file='PC3_1ball.mat'

#without gold samples
# path=r'C:\Users\Zhenya\Desktop\ELM2021\FLIM_code_10_July\Sample\1218_success'
# sample_file='PC_100_cycle_NoGold.mat'

# path=r'C:\Users\Zhenya\Desktop\ELM2021\FLIM_code_10_July\Sample\1218_success'
# sample_file='PC2_100_cycle_NoGold.mat'

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
if use_quan:
    PATH=r'./Q_addernet_pth/good_litemodel_loss_0.046051.pth'
    model = Quan_S_TauNet_AVE()
else:
    PATH=r'./addernet_pth/good_model_loss_0.049065.pth'
    model = S_TauNet_AVE()

#use cpu
if use_cpu==1:
    model.to(device="cpu")    
    model.eval()
    hist=torch.unsqueeze(hist,0)
    checkpoint=torch.load(PATH,map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint,strict=False)
    
    tau_amp=np.zeros(65536)
    tau_inten=np.zeros(65536)
    threshold=0
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
    threshold=0
    processed_pixels=0
    
    hist_nothre=hist_nor.unsqueeze(2)
    hist_nothre=hist_nothre.unsqueeze(2)
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

#power_end=nvmlDeviceGetPowerUsage(handle)
#print ("End Power:", power_end)
#
#power_consumpion=power_end-power_start
#print ("power_consumpion:", power_consumpion)
#
#device_util_rate=nvmlDeviceGetUtilizationRates(handle)
#print ("device_util_rate:", device_util_rate)
#
#nvmlShutdown()
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
path=r'C:\Users\Zhenya\Desktop\ELM2021\FLIM_code_10_July\beads samples'
sample_file='YG_GT.mat'
DataSet = io.loadmat(os.path.join(path,sample_file))
hist=DataSet.get('GT1')  
plt.imshow(hist, cmap='nipy_spectral')#nipy_spectral, gist_ncar,gist_stern
cb=plt.colorbar()
cb.ax.tick_params(labelsize=15)
plt.clim(0, 4)
plt.axis('off')
plt.show()
#%%
YG=3
Crism=2.1
tau_inten_filtered=tau_inten[tau_inten!=0]
tau_inten_mean=np.mean(tau_inten_filtered)
tau_inten_std=np.std(tau_inten_filtered)

abe_tau_inten=abs(tau_inten_filtered-YG)

accuracy=20*np.log10(np.mean(tau_inten_filtered/abe_tau_inten))
precision=20*np.log10(np.mean(tau_inten_filtered/tau_inten_std))

plt.imshow(tau_inten[1:251,5:256], cmap='nipy_spectral')#nipy_spectral, gist_ncar,gist_stern
cb=plt.colorbar()
cb.ax.tick_params(labelsize=15)
plt.clim(0, 4)
plt.axis('off')
plt.show()
savemat(r"./mat_result/YG_beads_AdderNet.mat", {"YG_beads_AdderNet":tau_inten[1:251,5:256]})

#%%plot CMM
path=r'C:\Users\Zhenya\Desktop\1DCNN_no_padding\S_TauNet_two_output\mat_result'
sample_file='YG_beads_CMM.mat'

DataSet = io.loadmat(os.path.join(path,sample_file))
hist=DataSet.get('tau_inten_ave_cmm')  

YG=3
Crism=2.1
tau_inten_filtered=hist[hist!=0]
tau_inten_mean=np.mean(tau_inten_filtered)
tau_inten_std=np.std(tau_inten_filtered)

abe_tau_inten=abs(tau_inten_filtered-YG)

accuracy=20*np.log10(np.mean(tau_inten_filtered/abe_tau_inten))
precision=20*np.log10(np.mean(tau_inten_filtered/tau_inten_std))

plt.imshow(hist[1:251,5:256], cmap='nipy_spectral')#nipy_spectral, gist_ncar,gist_stern
cb=plt.colorbar()
cb.ax.tick_params(labelsize=15)
plt.clim(0, 4)
plt.axis('off')
plt.show()
#%%plot NLSF
path=r'C:\Users\Zhenya\Desktop\1DCNN_no_padding\S_TauNet_two_output\mat_result'
sample_file='YG_beads_NLSF.mat'

DataSet = io.loadmat(os.path.join(path,sample_file))
hist=DataSet.get('tau_nlsf')  

YG=3
Crism=2.1
tau_inten_filtered=hist[hist!=0]
tau_inten_mean=np.mean(tau_inten_filtered)
tau_inten_std=np.std(tau_inten_filtered)

abe_tau_inten=abs(tau_inten_filtered-YG)

accuracy=20*np.log10(np.mean(tau_inten_filtered/abe_tau_inten))
precision=20*np.log10(np.mean(tau_inten_filtered/tau_inten_std))

plt.imshow(hist[1:251,5:256], cmap='nipy_spectral')#nipy_spectral, gist_ncar,gist_stern
cb=plt.colorbar()
cb.ax.tick_params(labelsize=13)
plt.clim(0, 4)
plt.axis('off')
plt.show()
#%%plot Mixed
path=r'C:\Users\Zhenya\Desktop\1DCNN_no_padding\S_TauNet_two_output\mat_result'
sample_file='mixed_beads_NLSF.mat'

DataSet = io.loadmat(os.path.join(path,sample_file))
hist=DataSet.get('tau_nlsf')  

plt.imshow(hist, cmap='nipy_spectral')#nipy_spectral, gist_ncar,gist_stern
cb=plt.colorbar()
cb.ax.tick_params(labelsize=15)
plt.clim(0, 3.5)
plt.axis('off')
plt.show()
#%%
from sklearn.metrics import mean_squared_error
path=r'C:\Users\Zhenya\Desktop\1DCNN_no_padding\S_TauNet_two_output\mat_result/'
sample_file='NLSF_100to1000_amp.mat'
DataSet = io.loadmat(os.path.join(path,sample_file))
hist=DataSet.get('tau2d_amp_lsq')  
hist_gt=DataSet.get('tau2d_amp_ave') 

mse_nlsf=mean_squared_error(hist,hist_gt)

plt.imshow(hist, cmap='gist_ncar')
plt.colorbar()
plt.clim(0, 3)
plt.axis('off')
plt.show()
#%%
# input = torch.randn(1,1,1,256).cuda()
# macs, params = profile(model, inputs=(input, ))
# summary(model,input)


