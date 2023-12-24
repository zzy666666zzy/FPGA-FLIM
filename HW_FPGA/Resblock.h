#include <ap_int.h>
#include <iostream>
#include <ap_fixed.h>
#include "parameter_define.h"

void Resblock (Dtype_f hist_in[Win_R1*Hin_R1*CHin_R1],
		Dtype_f feature_o[Wout_R2*Hout_R2*CHout_R2]
);

void R1_conv(Dtype_f feature_in[Win_R1*Hin_R1*CHin_R1],
		  Dtype_f feature_out[Wout_R1*Hout_R1*CHout_R1]
	);

void R2_conv(Dtype_f feature_in[Win_R2*Hin_R2*CHin_R2],
		  Dtype_f feature_out[Wout_R2*Hout_R2*CHout_R2]
	);

void Array_Add(Dtype_f Arr1[Wout_R1*Hout_R1*CHout_R1],
			   Dtype_f Arr2[Wout_R2*Hout_R2*CHout_R2],
			   Dtype_f Arr_added[Wout_R2*Hout_R2*CHout_R2]);

void Array_Add_1(Dtype_f Arr1[Wout_R1*Hout_R1*CHout_R1],
			   Dtype_f Arr2[Wout_R2*Hout_R2*CHout_R2],
			   Dtype_f Arr_added[Wout_R2*Hout_R2*CHout_R2]);
