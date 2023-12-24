# -*- coding: utf-8 -*-
#ZZY 17/Jan/2022

import os
import numpy as np
from parser_utils.conver_to_fix import Decimal_to_Binary
from parser_utils.conver_to_fix import float_to_fp


def write_fixed_binary(in_folder,out_folder,Data_Bit_Length,Weight_Int_Length,Weight_Frac_Length):
    g = os.walk(in_folder)  
    dec_float = np.zeros(1)
    for path,dir_list,file_list in g:  
        for file_name in file_list: #traverse each .txt file
            dec_file = open(in_folder+file_name, 'r') #open each .txt decimal parameter file
            dec = dec_file.read()
            dec_arr = dec.split()
            f = open(out_folder+file_name, 'w')
            for i in range(len(dec_arr)):
                dec_float=float(dec_arr[i])#conver str to float
                if dec_float > 2**(Weight_Int_Length-1):
                    dec_float = 2**(Weight_Int_Length-1)-2**(-Weight_Frac_Length)
                elif dec_float < -2**(Weight_Frac_Length-1):
                    dec_float = -2**(Weight_Frac_Length-1)
                bin_para = float_to_fp(dec_float,Weight_Int_Length,Weight_Frac_Length)
                f.write(bin_para+'\n')
                    