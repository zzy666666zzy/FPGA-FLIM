#include "xtoptop.h"
#include "platform.h"

//Set-value functions constructed from PL
void toptop_FixP(XToptop *InstancePtr,
		int hist_quantized1[],
		int hist_quantized2[],
		int hist_quantized3[],
		int hist_quantized4[],

		int Lifetimes_out1[],int Lifetimes_out2[],int Lifetimes_out3[],int Lifetimes_out4[])
{

	XToptop_Set_hist_in_1_V(InstancePtr,(unsigned int)hist_quantized1);
	XToptop_Set_hist_in_2_V(InstancePtr,(unsigned int)hist_quantized2);
	XToptop_Set_hist_in_3_V(InstancePtr,(unsigned int)hist_quantized3);
	XToptop_Set_hist_in_4_V(InstancePtr,(unsigned int)hist_quantized4);

	XToptop_Set_Lifetimes_out_1_V(InstancePtr,(unsigned int)Lifetimes_out1);
	XToptop_Set_Lifetimes_out_2_V(InstancePtr,(unsigned int)Lifetimes_out2);
	XToptop_Set_Lifetimes_out_3_V(InstancePtr,(unsigned int)Lifetimes_out3);
	XToptop_Set_Lifetimes_out_4_V(InstancePtr,(unsigned int)Lifetimes_out4);

	XToptop_Start(InstancePtr);
	while(!XToptop_IsDone(InstancePtr));
}
