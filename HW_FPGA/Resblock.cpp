#include "Resblock.h"
#include "Pre1.h"
#include <ap_int.h>
#include <iostream>
#include <ap_fixed.h>

void Array_Add(Dtype_f Arr1[Win_R1*Hin_R1*CHin_R1],
			   Dtype_f Arr2[Wout_R2*Hout_R2*CHout_R2],
			   Dtype_f Arr_added[Wout_R2*Hout_R2*CHout_R2])
{
	int loop_num=Wout_R1*Hout_R1*CHout_R1;
	Array_Add_label1:for(int i=0;i<loop_num;i++)
	{
		Arr_added[i]=Arr1[i]+Arr2[i];
		if(Arr_added[i]<0)//ReLu
		{
			Arr_added[i]=0;
		}
//	#ifndef __SYNTHESIS__
//	printf("Arr_added[%d]=%f\n",i,Arr_added[i]);
//	#endif
	}
}

void Resblock (Dtype_f hist_in[Win_R1*Hin_R1*CHin_R1],
		Dtype_f feature_out[Wout_R2*Hout_R2*CHout_R2]
)
{

Dtype_f feature_R1_o[Wout_R1*Hout_R1*CHout_R1];
Dtype_f feature_R2_o[Wout_R2*Hout_R2*CHout_R2];

R1_conv(hist_in,feature_R1_o);
R2_conv(feature_R1_o,feature_R2_o);

Array_Add(hist_in,feature_R2_o,feature_out);

}
