#include "AdderNet.h"
#include <iostream>
#include "Pre1.h"
#include "O1_3.h"
#include "Pre2.h"
#include "string.h"

void AdderNet_fixedp_5 (Dtype_f hist_in1[Win_Pre1*Hin_Pre1*CHin_Pre1],
		Dtype_f Lifetimes_out[2]
)
{
//IO declare AXI-4
//	#pragma HLS INTERFACE m_axi depth=4294967 port=hist_in1 offset=slave
//	#pragma HLS INTERFACE m_axi depth=4294967 port=hist_in2 offset=slave
//	#pragma HLS INTERFACE m_axi port=Lifetimes_out1 offset=slave
//	#pragma HLS INTERFACE m_axi port=Lifetimes_out2 offset=slave
//	#pragma HLS INTERFACE s_axilite port=return

#pragma HLS INLINE off

Dtype_f hist_buffer[Win_Pre1*Hin_Pre1*CHin_Pre1];
Dtype_f Lifetimes_buffer[2];

//memcpy(hist_buffer,hist_in1,Win_Pre1*Hin_Pre1*CHin_Pre1*sizeof(Dtype_f));//cache hist in buffer

Dtype_f feature_Pre1_o[Wout_Pre1*Hout_Pre1*CHout_Pre1];
Dtype_f feature_Pre2_o[Wout_Pre2*Hout_Pre2*CHout_Pre2];
Dtype_f feature_Resblock_o[Wout_R2*Hout_R2*CHout_R2];
Dtype_f feature_reshape[Wout_Pre2*Hout_Pre2*CHout_Pre2];


Dtype_f feature_O1_1_o[Wout_O1_1*Hout_O1_1*CHout_O1_1];
Dtype_f feature_O1_2_o[Wout_O1_2*Hout_O1_2*CHout_O1_2];

Dtype_f feature_O2_1_o[Wout_O2_1*Hout_O2_1*CHout_O2_1];
Dtype_f feature_O2_2_o[Wout_O2_2*Hout_O2_2*CHout_O2_2];

Dtype_f tau_inten[1];
Dtype_f tau_amp[1];

//*******AdderNet*******
Pre1_conv(hist_in1,feature_Pre1_o);
Pre2_conv(feature_Pre1_o,feature_Pre2_o);

Resblock(feature_Pre2_o,feature_Resblock_o);

for (int i=0;i<CHout_R2;i++)
	for(int j=0;j<Wout_R2;j++){
		#pragma HLS PIPELINE
		feature_reshape[j+i*Wout_R2]=feature_Resblock_o[j*CHout_R2+i];
//		#ifndef __SYNTHESIS__
//		printf("feature_reshape[%d]=%f\n",j+i*Wout_R2,feature_reshape[j+i*Wout_R2]);
//		#endif
	}

Dtype_f feature_reshape1[Wout_Pre2*Hout_Pre2*CHout_Pre2];
Dtype_f feature_reshape2[Wout_Pre2*Hout_Pre2*CHout_Pre2];

for(int i=0;i<140;i++)
{
	#pragma HLS PIPELINE
	feature_reshape1[i]=feature_reshape[i];
	feature_reshape2[i]=feature_reshape[i];
}

//Tau_inten output
O1_1(feature_reshape1,feature_O1_1_o);
O1_2(feature_O1_1_o,feature_O1_2_o);
O1_3(feature_O1_2_o,tau_inten);

//Tau_amp output
O2_1(feature_reshape2,feature_O2_1_o);
O2_2(feature_O2_1_o,feature_O2_2_o);
O2_3(feature_O2_2_o,tau_amp);

Lifetimes_out[0]=tau_inten[0];
Lifetimes_out[1]=tau_amp[0];
}
