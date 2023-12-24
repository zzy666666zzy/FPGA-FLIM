# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 17:53:11 2020

@author: pc
"""

import time,os
import torch
import numpy as np
import scipy.io as io
from scipy.io import savemat
import datetime
import matplotlib.pyplot as plt
import math

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import RMSprop,SGD
from torch.nn import MSELoss
import torch.nn.functional as F
from EarlyStopping import EarlyStopping

from S_TauNet_AVE_model import S_TauNet_AVE
from Q_S_TauNet_AVE_model import Quan_S_TauNet_AVE
from S_TauNet_AVE_model_logscaling import S_TauNet_AVE_loghist

import torch.optim.lr_scheduler as lr_scheduler

from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
#%%
#-----------------------------------------------------------------------------#
def load_data(path, data_name,test_ratio = 0.2,BATCH_SIZE = 128):
    DataSet = io.loadmat(os.path.join(path,data_name))
    indata=DataSet.get('Compressed_Hist_nor')
    indata=indata.reshape(indata.shape[0],1,indata.shape[1])
    
    tau_amp_ave=DataSet.get('tau_amp_total')
    tau_amp_ave=tau_amp_ave.reshape(-1)
    tau_inten_ave=DataSet.get('tau_inten_total')
    tau_inten_ave=tau_inten_ave.reshape(-1)

    targets = np.asarray([tau_amp_ave,tau_inten_ave])
    targets =targets.transpose()
    print('Input train-data size',indata.shape)
    print('Input train-data-label size',targets.shape)

    indata=np.expand_dims(indata, axis=1)
    indata = torch.from_numpy(indata)
    targets = torch.from_numpy(targets)
    torch_dataset = TensorDataset(indata,targets)
    test_size = round(test_ratio * len(indata))
    train_size =  len(indata) - test_size
    train, test = random_split(torch_dataset, [train_size,test_size])

    train_set = DataLoader(dataset=train,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=0)
    test_set = DataLoader(dataset=test,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=0)
    
    return train_set,test_set

def train_model(train_set, model,Quan,Epoch=300,USE_GPU = True, log_interval =50,patience=20):

    # define the optimization
    criterion = MSELoss().cuda()
    optimizer = RMSprop(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.993)
    
    # Record the loss 
    Train_Loss_Amp = list()
    Train_Loss_Inten = list()
    Train_Loss_Total = list()
    
    Val_Loss_Amp = list()
    Val_Loss_Inten = list()
    Val_Loss_Total = list()
    
    # Enumerate epochs
    start_time = time.time()
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    #Epoch loop
    for epoch in range(Epoch):
        print('Epoch {}/{}'.format(epoch+1, Epoch))
        
        # adjust_learning_rate(optimizer, epoch)
        for param_group in optimizer.param_groups:
            print('lr: ',param_group['lr'])
        
        #Do training, then validating
        for phase in ['train', 'val']:
            tic = time.time()
            if phase == 'train':
                model.train()  # Set model to training mode
                use_set = train_set
            else:
                model.eval()   # Set model to evaluating mode
                use_set = test_set
            # enumerate mini batches
            for batch_idx, (inputs, targets) in enumerate(use_set):
                #running loss
                l1 = list()
                l2 = list()
                l = list()
                
                if USE_GPU:
                    inputs, targets = inputs.cuda(), targets.cuda()
                # clear the gradients
                optimizer.zero_grad()
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    # compute the model output
                    yhat = model(inputs.float())
                    # calculate MSEloss
                    loss1 = criterion(yhat[0], targets[:,0].float())
                    loss2 = criterion(yhat[1], targets[:,1].float())
                    loss = loss1+loss2
                    
                
                if  phase == 'train':        
                    # credit assignment
                    loss.backward()
                    
                    # update model weights
                    optimizer.step()
                #Convert tensor to numpy
                l1.append(loss1.cpu().detach().numpy())
                l2.append(loss2.cpu().detach().numpy())
                l.append(loss.cpu().detach().numpy())
                
                
                if (batch_idx % log_interval == 0 and phase == 'train'):
                    print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                        epoch+1, batch_idx * len(inputs), len(use_set.dataset), 
                        100. * batch_idx / len(use_set), loss))
            
            if phase == 'train':
                Train_Loss_Amp.append(np.mean(l1))
                Train_Loss_Inten.append(np.mean(l2))
                Train_Loss_Total.append(np.mean(l))
                
            else:
                Val_Loss_Amp.append(np.mean(l1))
                Val_Loss_Inten.append(np.mean(l2))
                Val_Loss_Total.append(np.mean(l))
                
                early_stopping(epoch+1, np.mean(l), model, ckpt_dir) 
             
            toc = time.time()-tic
            
            scheduler.step()
            
            print('\n{} Loss: {:.6f} time: {:.4f}s'.format(phase, np.mean(l), toc))
            if phase == 'val':        
                print('-' * 50)
                
        if early_stopping.early_stop: 
            print("Early stopping")
            break    
        Record = {'Train_Loss_Amp':Train_Loss_Amp,
                  'Train_Loss_Inten':Train_Loss_Inten,
                  'Train_Loss_Total':Train_Loss_Total}
    
    stop_time = time.time() - start_time
    
    loss = {"Train_Loss_Amp": Train_Loss_Amp,"Train_Loss_Inten": Train_Loss_Inten,"Train_Loss_Total": Train_Loss_Total,
            "Val_Loss_Amp": Val_Loss_Amp,"Val_Loss_Inten": Val_Loss_Inten,"Val_Loss_Total": Val_Loss_Total}

    if Quan:
        savemat(r"./loss_quan_log_scaling.mat", loss)
    else:
        savemat(r"./loss_no_quan_log_scaling.mat", loss)
    
    print("Total training time: {:.2f}".format(stop_time))
    return Record
         
#%%   
#-----------------------------------------------------------------------------#
if __name__ == '__main__':  
    path = r'C:\Users\Zhenya\Desktop\ELM2021\FLIM_code_10_July\Output_Ave'   
    data_name = 'Synthetic_Train_mix_decays_log_scaling'
    ckpt_dir='./addernet_pth_log_scaling'
    os.makedirs(ckpt_dir,exist_ok=True)
    train_set, test_set = load_data(path, data_name)     
    USE_GPU = True

    Quan=0
    if Quan:
        model = Quan_S_TauNet_AVE()
    else:
        model = S_TauNet_AVE_loghist()

    if USE_GPU:
        model.cuda() 
    #paramter_initialize(model)
    Record = train_model(train_set, model,Quan,Epoch=300, USE_GPU = USE_GPU )    

    x = datetime.datetime.now()
    # model_data = dict(
    #     model = model.state_dict(), Record=Record,
    #     info = 'trained model with 350 epochs')
    # torch.save(model_data, 
    #         'S_Tau_model_{}_{}_{}_{}_{}.pth'
    #         .format(x.year,x.month, x.day,x.hour,x.minute),_use_new_zipfile_serialization=False)  
