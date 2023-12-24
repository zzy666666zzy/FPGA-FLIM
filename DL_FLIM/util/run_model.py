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

def run_model(folder_data,sample_file,pretrained_model,real_data_en,threshold):

    use_quan=0
    threshold=int(threshold)
    
    np.set_printoptions(suppress=True)
    torch.set_printoptions(precision=5)
    
    start = time.time()
    
    #---for complete data test-----
    #with gold samples
    # path=r'C:\Users\Zhenya\Desktop\S_Tau_Net_conference'
    # sample_file='PC_100_cycle.mat'
    
    # path=r'C:\Users\Zhenya\Desktop\ELM2021\FLIM_code_10_July\Sample'
    # sample_file='pc3_probe1.mat'
    
    # path=r'C:\Users\Zhenya\Desktop\ELM2021\FLIM_code_10_July\Sample'
    # sample_file='PC3_1ball.mat'
    
    #without gold samples
    # path=r'C:\Users\Zhenya\Desktop\ELM2021\FLIM_code_10_July\Sample\1218_success'
    # sample_file='PC_100_cycle_NoGold.mat'
    
    # path=r'C:\Users\Zhenya\Desktop\ELM2021\FLIM_code_10_July\Sample\1218_success'
    # sample_file='PC2_100_cycle_NoGold.mat'
    path=folder_data
    global hist_nor
    DataSet = io.loadmat(os.path.join(path,sample_file))
    if real_data_en:
        hist=DataSet.get('hist')
        hist_nor=DataSet.get('hist65536_nor')
    else:
        hist=DataSet.get('decay') 
        hist_nor=np.zeros((65536,256))
        for i in range(65536):
            hist_nor[i,:]=hist[i,:]/max(hist[i,:])
    
    hist=hist.astype(np.float32)
    hist=hist.reshape(hist.shape[0],1,hist.shape[1])
    hist = Variable(torch.from_numpy(hist))
    
    hist_nor=hist_nor.astype(np.float32)
    hist_nor=hist_nor.reshape(hist_nor.shape[0],1,hist_nor.shape[1])
    hist_nor = Variable(torch.from_numpy(hist_nor))
    
    #use gpu
    # if use_quan:
    #     PATH=r'./Q_addernet_pth/good_litemodel_loss_0.046051.pth'
    #     model = Quan_S_TauNet_AVE()
    # else:
    # PATH=r'./addernet_pth/ckpt_epoch_85_val_loss_0.032351.pth'
    model = S_TauNet_AVE()
    
    #use cpu
    #checkpoint = torch.load(PATH, map_location='cpu')
    #model.load_state_dict(checkpoint)
    
    checkpoint=torch.load(pretrained_model)
    model.load_state_dict(checkpoint,strict=False)
    model.eval()
    hist=torch.unsqueeze(hist,0)
    #%%
    tau_amp=np.zeros(65536)
    tau_inten=np.zeros(65536)
    processed_pixels=0
    
    hist_nothre=hist_nor.unsqueeze(2)
    hist_nothre=hist_nothre.unsqueeze(2)
    
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
    
    return tau_amp,tau_inten

def get_inten(path,sample_file,real_data_en):
    
    DataSet = io.loadmat(os.path.join(path,sample_file))
    if real_data_en:
        hist=DataSet.get('hist')  
    else:
        hist=DataSet.get('decay')  
    intensity=hist.reshape(256,256,256)
    intensity=intensity.sum(2)
    
    return intensity