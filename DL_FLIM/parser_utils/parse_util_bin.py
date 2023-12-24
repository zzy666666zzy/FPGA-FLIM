# -*- coding: utf-8 -*-

import numpy as np
import struct

def Record_Tensor_bin(filename_bin,tensor,name):
    print ("Recording tensor "+name+" ...")
    f = open(filename_bin+name+'.bin', 'wb')
    array=tensor;
    if(np.size(np.shape(array))==1):
        Record_Array1D(array,name,f)
    else:
        if(np.size(np.shape(array))==2):
            Record_Array2D(array,name,f)
        else:
            if(np.size(np.shape(array))==3):
                Record_Array3D(array,name,f)
            else:
                Record_Array4D(array,name,f)
    f.close();

def Record_Array1D(array,name,f):
    for i in range(np.shape(array)[0]):
        a=struct.pack('f',array[i])
        f.write(a)

def Record_Array2D(array,name,f):
    for j in range(np.shape(array)[1]):
        for i in range(np.shape(array)[0]):
            a=struct.pack('f',array[i][j])
            f.write(a)

def Record_Array3D(array,name,f):
    for k in range(np.shape(array)[2]):
        for j in range(np.shape(array)[1]):
            for i in range(np.shape(array)[0]):
                a=struct.pack('f',array[i][j][k])
                f.write(a)

def Record_Array4D(array,name,f):
    for l in range(np.shape(array)[3]):
        for k in range(np.shape(array)[2]):
            for j in range(np.shape(array)[1]):
                for i in range(np.shape(array)[0]):
                    a=struct.pack('f',array[i][j][k][l])
                    f.write(a)

