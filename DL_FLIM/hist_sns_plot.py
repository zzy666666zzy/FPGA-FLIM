# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import scipy.io as io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load NLSF
path=r'C:\Users\Zhenya\Desktop\1DCNN_no_padding\S_TauNet_two_output\mat_result'
sample_file='mixed_beads_NLSF'
DataSet = io.loadmat(os.path.join(path,sample_file))
NLSF=DataSet.get('tau_nlsf')
# Load CMM
path=r'C:\Users\Zhenya\Desktop\1DCNN_no_padding\S_TauNet_two_output\mat_result'
sample_file='mixed_beads_CMM'
DataSet = io.loadmat(os.path.join(path,sample_file))
CMM=DataSet.get('tau_cmm')
# Load CNN
path=r'C:\Users\Zhenya\Desktop\1DCNN_no_padding\S_TauNet_two_output\mat_result'
sample_file='Mixed_CNN'
DataSet = io.loadmat(os.path.join(path,sample_file))
One_DCNN=DataSet.get('Mixed_CNN')
# Load FLAN
path=r'C:\Users\Zhenya\Desktop\1DCNN_no_padding\S_TauNet_two_output\mat_result'
sample_file='mixed_flan.mat'
DataSet = io.loadmat(os.path.join(path,sample_file))
FLAN=DataSet.get('tau_flan')
# Load FLAN+LS
path=r'C:\Users\Zhenya\Desktop\1DCNN_no_padding\S_TauNet_two_output\mat_result'
sample_file='Mixed_beads_AdderNet_LS'
DataSet = io.loadmat(os.path.join(path,sample_file))
FLAN_LS=DataSet.get('Mixed_beads_AdderNet_LS')

# Plot
sns.set_style("ticks")
kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':3})

plt.figure(figsize=(10,7), dpi= 90)
sns.set(font_scale = 2)
sns.distplot(NLSF, color="dodgerblue", label="NLSF",hist=False,**kwargs)
sns.distplot(CMM, color="orange", label="CMM",hist=False,**kwargs)
sns.distplot(One_DCNN, color="deeppink", label="CNN",hist=False,**kwargs)
sns.distplot(FLAN, color="yellowgreen", label="FLAN",hist=False,**kwargs)
sns.distplot(FLAN_LS, color="crimson", label="FLAN+LS",hist=False,**kwargs)
plt.xlim(0.3,3.2)
plt.ylim(0,0.25)
plt.legend()
plt.legend(loc='upper left')

