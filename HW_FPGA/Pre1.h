
#include <ap_int.h>
#include <iostream>
#include <ap_fixed.h>
#include "parameter_define.h"

//typedef float Dtype_para;
//typedef float Dtype_f;
//typedef float Dtype_w;
//typedef float Dtype_mul;
//typedef float Dtype_acc;

void Pre1_conv(Dtype_f_pre1_i feature_in[Win_Pre1*Hin_Pre1*CHin_Pre1],
		Dtype_f_pre1_o feature_out[Wout_Pre1*Hout_Pre1*CHout_Pre1]
	);

