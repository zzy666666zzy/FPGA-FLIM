# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 17:49:23 2020

@author: pc
"""

import torch
import numpy as np
import torch.nn as nn
from torch.nn import Conv1d, BatchNorm2d
from torch.nn import ReLU,Sigmoid
from torch.nn import Module,Sequential
from torch.nn.init import xavier_normal_
import torch.nn.functional as F
from adder_basic import adder2d

def Conv_Add(in_planes, out_planes, kernel1_size, kernel2_size, stride, padding=0, bias=False):
    return adder2d(in_planes, out_planes, kernel1_size, kernel2_size,stride, padding=padding, bias=False)


class ResBlock(Module):
    '''
    submodule: Residual Block
    '''
    def __init__(self, inchannel, outchannel, kernel_size1=1,kernel_size2=1,stride=1, 
                 input_length=80):
        super(ResBlock, self).__init__()
        self.left = Sequential(
                Conv_Add(inchannel,outchannel,kernel_size1,kernel_size2,stride),
                BatchNorm2d(outchannel),
                ReLU(),
                Conv_Add(inchannel,outchannel,kernel_size1,kernel_size2,stride),
                BatchNorm2d(outchannel) )

    def forward(self, x):
        out = self.left(x)
        residual = x 
        out += residual
        out = ReLU()(out)
        return out
    
class S_TauNet_AVE_loghist(Module):
    '''
    Model for calculating the fluorescence lifetimes
    '''
    def __init__(self):
        super(S_TauNet_AVE_loghist, self).__init__()
        self.PreRes1_conv = Conv_Add(1,5,1,7,3)
        self.PreRes1_bn = BatchNorm2d(5)
        
        self.PreRes2_conv = Conv_Add(5,10,1,5,3)
        self.PreRes2_bn = BatchNorm2d(10)
            
        self.R1 = ResBlock(10,10)
        
        self.O1_1_conv =  Conv_Add(70,30,1,1,1)
        self.O1_1_bn = BatchNorm2d(30)
        self.O1_2_conv = Conv_Add(30,1,1,1,1)
        self.O1_2_bn = BatchNorm2d(1)
        
        self.O2 = Sequential(Conv_Add(70,30,1,1,1),
                             BatchNorm2d(30),
                             ReLU(),
                             Conv_Add(30,1,1,1,1),
                             BatchNorm2d(1),
                             ReLU())
        
        
    # forward propagate input
    def forward(self, x):
        x = self.PreRes1_conv(x)
        x = self.PreRes1_bn(x)
        x = F.relu(x)
        x = self.PreRes2_conv(x)
        x = self.PreRes2_bn(x)
        x = F.relu(x)
        x = self.R1(x)
        
        x = x.view(x.size(0),-1,1)
        x = x.unsqueeze(2)

        x_O1 = self.O1_1_conv(x)
        x_O1 = self.O1_1_bn(x_O1)
        x_O1 = F.relu(x_O1)
        x_O1 = self.O1_2_conv(x_O1)
        x_O1 = self.O1_2_bn(x_O1)
        x_O1 = F.relu(x_O1)
        tau_amp_ave = x_O1
        tau_amp_ave = tau_amp_ave.view(-1)
        tau_inten_ave = self.O2(x)
        tau_inten_ave = tau_inten_ave.view(-1)
        
        return [tau_amp_ave,tau_inten_ave]
    