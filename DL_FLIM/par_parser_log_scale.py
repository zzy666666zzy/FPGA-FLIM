#ZZY 02/June/2021
import torch
import os
import numpy as np
from S_TauNet_AVE_model import S_TauNet_AVE
from Q_S_TauNet_AVE_model import Quan_S_TauNet_AVE
import numpy as np
from parser_utils.parse_util import Record_Tensor_txt
from parser_utils.parse_util_bin import Record_Tensor_bin
import struct
from parser_utils.write_binary import write_fixed_binary
from S_TauNet_AVE_model_logscaling import S_TauNet_AVE_loghist

np.set_printoptions(suppress=True)
torch.set_printoptions(precision=7)

#output bin
out_bin=0

np.set_printoptions(threshold=np.inf)
model = S_TauNet_AVE_loghist()
param={} #print size of each layer
for name,parameters in model.state_dict().items():
    print(name,':',parameters.size())
    param[name]=parameters.cpu().detach().numpy()
#%%
#print all parameters
pth='addernet_pth_log_scaling/S_TauNet_AVE_train_log_scaling_val_loss_0.068184.pth'
model.load_state_dict(torch.load(pth))
for i in model.named_parameters():
    print(i)
#%% this module for Weights and bias 
filename_bin=r'./para_BN_W&b_bin_log_scale/'
filename_txt=r'./para_BN_W&b_txt_log_scale/'
def Params_Extractor(name,tensor):
  tensor_tmp = param[tensor]
  if out_bin==1:
      Record_Tensor_bin(filename_bin,tensor_tmp,name)
  else:
      Record_Tensor_txt(filename_txt,tensor_tmp,name)
#%% this module for BN parameters
filename_bin=r'./para_BN_W&b_bin_log_scale/'
filename_txt=r'./para_BN_W&b_txt_log_scale/'
def Scale_Shift (mean,var,gamma,beta,name_scale,name_shift,eps=0.00001):
    tensor_mean = param[mean]
    tensor_var = param[var]
    tensor_gamma = param[gamma]
    tensor_beta = param[beta]

    coeff_A=tensor_gamma/np.sqrt(tensor_var+eps)
    coeff_B=tensor_beta-(tensor_mean*tensor_gamma)/np.sqrt(tensor_var+eps)
    #-------Scale-------
    coeff_A = coeff_A.reshape(-1,1) #make tensor to vector
    if out_bin==1:
        f = open(filename_bin+name_scale+'.bin', 'wb')
        for i in range(np.shape(coeff_A)[0]):#Only one dimension
            a=struct.pack('f',coeff_A[i]) #convert list to string representation
            f.write(a)
    #Write to txt anyway
    coeff_A = np.array2string(coeff_A,separator='  ',suppress_small=True)
    coeff_A=coeff_A.replace('[','').replace(']','')
    with open(filename_txt+name_scale+'.txt', 'w') as f:  
        f.write(coeff_A) 
    #-------Shift-------
    coeff_B = coeff_B.reshape(-1,1) #make tensor to vector
    if out_bin==1:
        f = open(filename_bin+name_shift+'.bin', 'wb')
        for i in range(np.shape(coeff_B)[0]): #Only one dimension
            a=struct.pack('f',coeff_B[i])
            f.write(a)
    #Write to txt anyway
    coeff_B = np.array2string(coeff_B,separator='  ',suppress_small=True)
    coeff_B=coeff_B.replace('[','').replace(']','')
    with open(filename_txt+name_shift+'.txt', 'w') as f:  
        f.write(coeff_B) 

#%% Extract parameters in batch normalization
#PreRes--------------------------------------------------
Params_Extractor("W_PR1"        ,"PreRes1_conv.Weight")
Scale_Shift("PreRes1_bn.running_mean","PreRes1_bn.running_var","PreRes1_bn.weight","PreRes1_bn.bias",
            "Sc_PR1","Sh_PR1")

Params_Extractor("W_PR2"        ,"PreRes2_conv.Weight")
Scale_Shift("PreRes2_bn.running_mean","PreRes2_bn.running_var","PreRes2_bn.weight","PreRes2_bn.bias",
            "Sc_PR2","Sh_PR2")
#Resblock------------------------------------------------                 
Params_Extractor("W_R0"       ,"R1.left.0.Weight")
Scale_Shift("R1.left.1.running_mean","R1.left.1.running_var","R1.left.1.weight","R1.left.1.bias",
            "Sc_R1","Sh_R1")
                                           
Params_Extractor("W_R3"       ,"R1.left.3.Weight")
Scale_Shift("R1.left.4.running_mean","R1.left.4.running_var","R1.left.4.weight","R1.left.4.bias",
            "Sc_R4","Sh_R4")
#Tau_A-------------------------------------------------------             
Params_Extractor("W_O11"            ,"O1_1_conv.Weight")
Scale_Shift("O1_1_bn.running_mean","O1_1_bn.running_var","O1_1_bn.weight","O1_1_bn.bias",
            "Sc_O11","Sh_O11")
                                           
Params_Extractor("W_O12"            ,"O1_2_conv.Weight")
Scale_Shift("O1_2_bn.running_mean","O1_2_bn.running_var","O1_2_bn.weight","O1_2_bn.bias",
            "Sc_O12","Sh_O12")
                                           
#Tau_I--------------------------------------------------------
Params_Extractor("W_O20"            ,"O2.0.Weight")
Scale_Shift("O2.1.running_mean","O2.1.running_var","O2.1.weight","O2.1.bias",
            "Sc_O21","Sh_O21")
                                           
Params_Extractor("W_O22"            ,"O2.3.Weight")
Scale_Shift("O2.4.running_mean","O2.4.running_var","O2.4.weight","O2.4.bias",
            "Sc_O22","Sh_O22")
                                          
#%% ZZY 17/Jan/2022 
#Convert decimal to binary, and write to files
#get floating-point number first (in_folder) and convert them to fixed-point (out_folder)

in_folder=r'./para_BN_W&b_txt/'
out_folder=r'./para_W_extracted_txt/'#final results

Data_Bit_Length=16
Weight_Int_Length=8
Weight_Frac_Length=Data_Bit_Length-Weight_Int_Length
                    
#Convert to binary datatype            
write_fixed_binary(in_folder,out_folder,Data_Bit_Length,Weight_Int_Length,Weight_Frac_Length)
