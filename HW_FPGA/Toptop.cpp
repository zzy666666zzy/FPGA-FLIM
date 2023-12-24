
#include "Toptop.h"
#include <iostream>
#include "Pre1.h"
#include "AdderNet.h"
#include "Toptop.h"

void TOPtop_four_core (Dtype_f_pre1_i hist_in_1[Win_Pre1*Hin_Pre1*CHin_Pre1],
		Dtype_f_pre1_i hist_in_2[Win_Pre1*Hin_Pre1*CHin_Pre1],
		Dtype_f_pre1_i hist_in_3[Win_Pre1*Hin_Pre1*CHin_Pre1],
		Dtype_f_pre1_i hist_in_4[Win_Pre1*Hin_Pre1*CHin_Pre1],
			 //Dtype_f hist_in_5[Win_Pre1*Hin_Pre1*CHin_Pre1],
		Dtype_f Lifetimes_out_1[2],
		Dtype_f Lifetimes_out_2[2],
		Dtype_f Lifetimes_out_3[2],
		Dtype_f Lifetimes_out_4[2]
		//Dtype_f Lifetimes_out_5[2],
		)
{

#pragma HLS INTERFACE m_axi depth=256 port=hist_in_1 offset=slave
#pragma HLS INTERFACE m_axi depth=256 port=hist_in_2 offset=slave
#pragma HLS INTERFACE m_axi depth=256 port=hist_in_3 offset=slave
#pragma HLS INTERFACE m_axi depth=256 port=hist_in_4 offset=slave
//#pragma HLS INTERFACE m_axi depth=256 port=hist_in_5 offset=slave

#pragma HLS INTERFACE m_axi port=Lifetimes_out_1 offset=slave
#pragma HLS INTERFACE m_axi port=Lifetimes_out_2 offset=slave
#pragma HLS INTERFACE m_axi port=Lifetimes_out_3 offset=slave
#pragma HLS INTERFACE m_axi port=Lifetimes_out_4 offset=slave
//#pragma HLS INTERFACE m_axi port=Lifetimes_out_5 offset=slave

#pragma HLS INTERFACE s_axilite port=return

Dtype_f_pre1_i hist_buffer1[Win_Pre1*Hin_Pre1*CHin_Pre1];
Dtype_f_pre1_i hist_buffer2[Win_Pre1*Hin_Pre1*CHin_Pre1];
Dtype_f_pre1_i hist_buffer3[Win_Pre1*Hin_Pre1*CHin_Pre1];
Dtype_f_pre1_i hist_buffer4[Win_Pre1*Hin_Pre1*CHin_Pre1];
//Dtype_f hist_buffer5[Win_Pre1*Hin_Pre1*CHin_Pre1];

memcpy(hist_buffer1,hist_in_1,256*sizeof(Dtype_f));
memcpy(hist_buffer2,hist_in_2,256*sizeof(Dtype_f));
memcpy(hist_buffer3,hist_in_3,256*sizeof(Dtype_f));
memcpy(hist_buffer4,hist_in_4,256*sizeof(Dtype_f));
//memcpy(hist_buffer5,hist_in_4,256*sizeof(Dtype_f));

Dtype_f Lifetimes_buffer1[2],Lifetimes_buffer2[2],Lifetimes_buffer3[2],Lifetimes_buffer4[2];//,Lifetimes_buffer5[2];

AdderNet_fixedp_1 ( hist_buffer1, Lifetimes_buffer1);
AdderNet_fixedp_2 ( hist_buffer2, Lifetimes_buffer2);
AdderNet_fixedp_3 ( hist_buffer3, Lifetimes_buffer3);
AdderNet_fixedp_4 ( hist_buffer4, Lifetimes_buffer4);
//AdderNet_fixedp_5 ( hist_buffer5, Lifetimes_buffer5);

memcpy(Lifetimes_out_1,Lifetimes_buffer1,2*sizeof(Dtype_f));
memcpy(Lifetimes_out_2,Lifetimes_buffer2,2*sizeof(Dtype_f));
memcpy(Lifetimes_out_3,Lifetimes_buffer3,2*sizeof(Dtype_f));
memcpy(Lifetimes_out_4,Lifetimes_buffer4,2*sizeof(Dtype_f));
//memcpy(Lifetimes_out_5,Lifetimes_buffer4,2*sizeof(Dtype_f));
}
