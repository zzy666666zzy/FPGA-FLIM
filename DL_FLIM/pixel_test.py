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
from torchsummary import summary
from torchstat import stat
from thop import profile
from S_TauNet_AVE_model_logscaling import S_TauNet_AVE_loghist
from pynvml import *
    
#%%
use_quan=0

np.set_printoptions(suppress=True)
torch.set_printoptions(precision=5)

path=r'C:\Users\Zhenya\Desktop\ELM2021\FLIM_code_10_July\Sample'
sample_file='pc3_probe1.mat'


hist_nor=[0,0,0,0,0,0,0,0,0,0,0.0292887,0.083682,0.2468619,0.5230126,0.6485356,0.7949791,0.9665272,0.9832636,0.9790795,1,0.8870293,0.9246862,0.7112971,0.6987448,0.6694561,0.7866109,0.6987448,0.6820084,0.707113,0.6694561,0.6359833,0.5606695,0.5439331,0.5439331,0.6276151,0.623431,0.5690377,0.5020921,0.4811715,0.3933054,0.4686192,0.5230126,0.3556485,0.3974895,0.4435146,0.4016736,0.3263598,0.3849372,0.3221757,0.3012552,0.3598326,0.2970711,0.3389121,0.2552301,0.2803347,0.3054393,0.2384937,0.3012552,0.2803347,0.3263598,0.2301255,0.2343096,0.209205,0.2426778,0.2050209,0.2384937,0.1966527,0.2217573,0.1799163,0.209205,0.167364,0.1882845,0.1171548,0.1548117,0.1297071,0.1799163,0.125523,0.1213389,0.1338912,0.1213389,0.1422594,0.1297071,0.125523,0.125523,0.083682,0.1338912,0.125523,0.0878661,0.0920502,0.1129707,0.0920502,0.1171548,0.1046025,0.0585774,0.0794979,0.1004184,0.0585774,0.0711297,0.0585774,0.1046025,0.0585774,0.0878661,0.0962343,0.083682,0.0753138,0.0669456,0.0543933,0.083682,0.0669456,0.0460251,0.0543933,0.1087866,0.0794979,0.0585774,0.0711297,0.0585774,0.0585774,0.0585774,0.0502092,0.0502092,0.0460251,0.0502092,0.0543933,0.0543933,0.0585774,0.0460251,0.0334728,0.0460251,0.041841,0.0334728,0.0502092,0.0167364,0.0251046,0.041841,0.0292887,0.0376569,0.041841,0.0543933,0.0251046,0.0167364,0.0209205,0.041841,0.0251046,0.0167364,0.0376569,0.0292887,0.0251046,0.0209205,0.0251046,0.0251046,0.0083682,0.0167364,0.0083682,0.0125523,0.0251046,0.0292887,0.0292887,0.0083682,0.0167364,0.0167364,0.0292887,0.0125523,0.0041841,0.0167364,0.0167364,0.0041841,0.0209205,0,0.0167364,0.0125523,0.0125523,0.0125523,0,0.0125523,0,0.0083682,0.0125523,0.0083682,0.0167364,0.0041841,0.0125523,0.0125523,0.0041841,0.0083682,0.0125523,0,0.0041841,0.0125523,0,0.0083682,0.0041841,0.0125523,0.0083682,0.0167364,0.0083682,0.0041841,0.0167364,0.0083682,0.0041841,0.0041841,0,0.0041841,0.0167364,0,0,0.0041841,0,0.0041841,0,0,0.0041841,0.0041841,0.0041841,0.0041841,0.0167364,0.0041841,0.0125523,0,0,0.0041841,0,0,0,0,0.0041841,0,0,0.0041841,0.0041841,0,0.0083682,0.0041841,0,0,0,0,0,0.0083682,0,0,0.0041841,0,0,0.0041841,0.0041841,0,0,0,0,0,0,0,0,0,0,0.0041841
  ]

hist_nor = Variable(torch.from_numpy(np.array(hist_nor)).float())
hist_nor = hist_nor.unsqueeze(0).unsqueeze(0).unsqueeze(0)

PATH=r'./addernet_pth/good_model_loss_0.049065.pth'
model = S_TauNet_AVE()

checkpoint=torch.load(PATH)
model.load_state_dict(checkpoint,strict=False)
model.eval()
start = time.time()
with torch.no_grad():
    yhat = model(hist_nor)
    
end = time.time()
time_consumption=end - start
#%% GPU different batch sizes
np.set_printoptions(suppress=True)
torch.set_printoptions(precision=5)

path=r'C:\Users\Zhenya\Desktop\ELM2021\FLIM_code_10_July\Output_Ave'
sample_file='Synthetic_2D_Pattern_1000to5000'

DataSet = io.loadmat(os.path.join(path,sample_file))
hist=DataSet.get('Compressed_Hist_nor')  
# hist_log_scaling=DataSet.get('Compressed_Hist') 

hist=hist.astype(np.float32)
hist=hist.reshape(hist.shape[0],1,hist.shape[1])

PATH='S_TauNet_AVE_train_log_scaling.py_val_loss_0.038527.pth'
model = S_TauNet_AVE_loghist()
checkpoint=torch.load(PATH)
model.load_state_dict(checkpoint,strict=False)
model.eval()

start = time.time()

hist_tensor = Variable(torch.from_numpy(hist))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
hist_tensor = hist_tensor.to(device)


hist_adder=np.zeros((65536,80))
hist_cuda = Variable(torch.from_numpy(hist_adder))
hist_cuda.to(device)

hist_cuda=hist_tensor.unsqueeze(1)
    
start = time.time()
with torch.no_grad():
    yhat = model(hist_cuda)

end = time.time()
print(f"Runtime of the program is {end - start}")

#%% CPU different batch sizes
np.set_printoptions(suppress=True)
torch.set_printoptions(precision=5)

path=r'C:\Users\Zhenya\Desktop\ELM2021\FLIM_code_10_July\Output_Ave'
sample_file='four_pixels'

DataSet = io.loadmat(os.path.join(path,sample_file))
hist=DataSet.get('Compressed_Hist_nor_4')  
# hist_log_scaling=DataSet.get('Compressed_Hist') 

hist=hist.astype(np.float32)
hist=hist.reshape(hist.shape[0],1,hist.shape[1])

PATH='S_TauNet_AVE_train_log_scaling.py_val_loss_0.038527.pth'
model = S_TauNet_AVE_loghist()
checkpoint=torch.load(PATH)
model.load_state_dict(checkpoint,strict=False)
model.eval()

start = time.time()

hist_tensor = Variable(torch.from_numpy(hist))

device = torch.device('cpu')
model.to(device)
hist_tensor = hist_tensor.to(device)


hist_adder=np.zeros((65536,80))
hist_cuda = Variable(torch.from_numpy(hist_adder))
hist_cuda.to(device)

hist_cuda=hist_tensor.unsqueeze(1)
    
start = time.time()
with torch.no_grad():
    yhat = model(hist_cuda)

end = time.time()
print(f"Runtime of the program is {end - start}")