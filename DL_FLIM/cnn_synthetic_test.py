# -*- coding: utf-8 -*-
#ZZY 02/June/2021

import os
import torch
import numpy as np
import scipy.io as sio
from S_TauNet_AVE_model import S_TauNet_AVE
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.io import savemat
import time
from torchsummary import summary
from thop import profile
from sklearn.metrics import mean_squared_error
import scipy.io as io
from CNN_Two_o import S_Tau_Net_Two_o
#%%
start = time.time()

path=r'C:\Users\Zhenya\Desktop\ELM2021\FLIM_code_10_July\Output_Ave'
sample_file='Synthetic_2D_Pattern_1000to5000'

DataSet = io.loadmat(os.path.join(path,sample_file))
hist=DataSet.get('decay')  
tau2d_amp_ave_gt=DataSet.get('tau2d_amp_ave')  
tau2d_inten_ave_gt=DataSet.get('tau2d_inten_ave')  

hist=hist.astype(np.float32)
hist=hist.reshape(hist.shape[0],1,hist.shape[1])

hist = Variable(torch.from_numpy(hist))

#use gpu
# PATH=r'./addernet_pth/good_model_loss_0.018485_bi_40000_mono100_mixed_Np10to100.pth'
# model = S_TauNet_AVE()
#use cpu
#checkpoint = torch.load(PATH, map_location='cpu')
#model.load_state_dict(checkpoint)
# use cnn
PATH=r'./CNN_val_loss_0.060866.pth'
model = S_Tau_Net_Two_o()

checkpoint=torch.load(PATH)
model.load_state_dict(checkpoint,strict=False)
model.eval()
hist=torch.unsqueeze(hist,0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
hist = hist.to(device)

tau_amp=np.zeros(65536)
tau_inten=np.zeros(65536)
threshold=0

with torch.no_grad():
    for i in range(65536):
        hist_thre=torch.squeeze(torch.squeeze(hist))
        if torch.sum(hist_thre[i,:],dim=0)<threshold:
            yhat=0
        else:
            hist_adder = hist[:,i,:,:]/torch.max(hist[:,i,:,:])
            yhat = model(hist_adder)
            tau_amp[i]=yhat[0]
            tau_inten[i]=yhat[1]
        if (i%10000==0):
            print('{} pixel processed'.format(i))

tau_amp=tau_amp.reshape(256,256)
tau_inten=tau_inten.reshape(256,256)

tau_amp[np.isnan(tau_amp)] = 0
tau_inten[np.isnan(tau_inten)] = 0

end = time.time()

print('------Inference Done------')
print(f"Runtime of the program is {end - start}")

sio.savemat(r"./mat_result/cnn_1000to5000_amp.mat", {"cnn_1000to5000_amp":tau_amp})
sio.savemat(r"./mat_result/cnn_1000to5000_inten.mat", {"cnn_1000to5000_inten":tau_inten})
#%%
tau_amp=np.transpose(tau_amp,[1,0])
plt.imshow(tau_amp, cmap='gist_ncar')
plt.colorbar()
plt.clim(0, 3)
plt.axis('off')
plt.show()
#%%
tau_inten=np.transpose(tau_inten,[1,0])
plt.imshow(tau_inten, cmap='gist_ncar')
plt.colorbar()
plt.clim(0, 3)
plt.axis('off')
plt.show()
#%%
mse_amp=mean_squared_error(tau_amp,tau2d_amp_ave_gt)
mse_inten=mean_squared_error(tau_inten,tau2d_inten_ave_gt)

# input = torch.randn(1,1,1,256).cuda()
# macs, params = profile(model, inputs=(input, ))
# summary(model,input)
#%%Real beads test
path=r'C:\Users\Zhenya\Desktop\ELM2021\FLIM_code_10_July\beads samples'
sample_file='beads_100c_p6.mat'

DataSet = io.loadmat(os.path.join(path,sample_file))
hist=DataSet.get('beads_100c_p6')  
hist_nor=DataSet.get('hist65536_nor')

hist=hist.astype(np.float32)
hist=hist.reshape(hist.shape[0],1,hist.shape[1])
hist = Variable(torch.from_numpy(hist))

hist_nor=hist_nor.astype(np.float32)
hist_nor=hist_nor.reshape(hist_nor.shape[0],1,hist_nor.shape[1])
hist_nor = Variable(torch.from_numpy(hist_nor)) 


#use gpu
# PATH=r'./addernet_pth/good_model_loss_0.018485_bi_40000_mono100_mixed_Np10to100.pth'
# model = S_TauNet_AVE()
#use cpu
#checkpoint = torch.load(PATH, map_location='cpu')
#model.load_state_dict(checkpoint)
# use cnn
PATH=r'./CNN_val_loss_0.060866.pth'
model = S_Tau_Net_Two_o()

device = torch.device("cuda")
model.eval()
model=model.cuda()
hist=torch.unsqueeze(hist,0)
checkpoint=torch.load(PATH,map_location="cuda:0")
model.load_state_dict(checkpoint,strict=False)

tau_amp=np.zeros(65536)
tau_inten=np.zeros(65536)
threshold=2000
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
sio.savemat(r"./mat_result/YG_beads_CNN.mat", {"YG_beads_CNN":tau_inten[1:251,5:256]})
#%%
plt.imshow(tau_inten, cmap='nipy_spectral')#nipy_spectral, gist_ncar,gist_stern
cb=plt.colorbar()
cb.ax.tick_params(labelsize=15)
plt.clim(0, 3.5)
plt.axis('off')
plt.show()
sio.savemat(r"./mat_result/Mixed_CNN.mat", {"Mixed_CNN":tau_inten})
