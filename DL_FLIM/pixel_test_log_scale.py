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

use_quan=0

np.set_printoptions(suppress=True)
torch.set_printoptions(precision=5)

start = time.time()

path=r'C:\Users\Zhenya\Desktop\ELM2021\FLIM_code_10_July\Sample'

hist_nor=[0,0,0.0076336,0.022901,0.015267,0,0.0076336,0.0076336,0.0076336,0.030534,0.16031,0.48855,1,0.43511,0.47328,0.23664,0.17557,0.22137,0.16031,0.12214,0.18321,0.14504,0.091603,0.12977,0.099237,0.14504,0.099237,0.12214,0.10687,0.091603,0.099237,0.15267,0.17557,0.15267,0.053435,0.091603,0.21374,0.12214,0.091603,0.1145,0.099237,0.076336,0.083969,0.068702,0.053435,0.045802,0.099237,0.053435,0.068702,0.10687,0.030534,0.053435,0.068702,0.061069,0.045802,0.030534,0.091603,0.068702,0.038168,0.053435,0.061069,0.068702,0.053435,0.030534,0.030534,0.015267,0.038168,0.061069,0.061069,0.0076336,0.061069,0.053435,0.068702,0.0076336,0.015267,0.015267,0.022901,0.053435,0.076336,0.022901
]

hist_nor = Variable(torch.from_numpy(np.array(hist_nor)).float())
hist_nor = hist_nor.unsqueeze(0).unsqueeze(0).unsqueeze(0)

PATH=r'addernet_pth_log_scaling/S_TauNet_AVE_train_log_scaling_val_loss_0.068184.pth'
model = S_TauNet_AVE_loghist()

checkpoint=torch.load(PATH)
model.load_state_dict(checkpoint,strict=False)
model.eval()
#%%
with torch.no_grad():
    yhat = model(hist_nor)



