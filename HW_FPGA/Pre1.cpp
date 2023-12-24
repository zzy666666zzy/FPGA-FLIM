#include "Pre1.h"
#include "ap_int.h"

static Dtype_para W[Kx_Pre1*Ky_Pre1*CHin_Pre1*CHout_Pre1]={
		#include "./parameters/Pre1_weight.h"
		};
static Dtype_para scale[CHout_Pre1]={
		#include "./parameters/Pre1_scale.h"
		};
static Dtype_para shift[CHout_Pre1]={
		#include "./parameters/Pre1_shift.h"
		};

void Pre1_conv(
		Dtype_f_pre1_i feature_in[Win_Pre1*Hin_Pre1*CHin_Pre1],
		Dtype_f_pre1_o feature_out[Wout_Pre1*Hout_Pre1*CHout_Pre1]
	)
{
	#pragma HLS ARRAY_PARTITION variable=W complete
	#pragma HLS ARRAY_PARTITION variable=scale complete
	#pragma HLS ARRAY_PARTITION variable=shift complete
	//#pragma HLS ARRAY_PARTITION variable=feature_in complete

	for(int cout=0;cout<CHout_Pre1;cout++)
		for(int i=0;i<Hout_Pre1;i++)
			for(int j=0;j<Wout_Pre1;j++)
			{
				#pragma HLS PIPELINE
				//Above consume 245 (49*5) cycles to finish (Trip counts)
				//Below consume 7 cycles to finish (internal iteration)
				Dtype_f_pre1_o sum=0;
				for(int ii=0;ii<Ky_Pre1;ii++)
					for(int jj=0;jj<Kx_Pre1;jj++)
					{
						ap_int<16> h=i*Sy_Pre1+ii;
						ap_int<16> w=j*Sx_Pre1+jj;
						if(h>=0 && w>=0 && h<Hin_Pre1 && w<Win_Pre1)
						{
							for(int cin=0;cin<CHin_Pre1;cin++)
							{
								//Feature [H][W][C]
								//kernel: [Ky][Kx][CHin][CHout]
								//Dtype_mul tp=feature_in[h][w][cin]*w[ii][jj][cin][cout];
								//std::cout<<"h:"<<h<<",w"<<w<<",cin"<<cin<<"\n";
								//std::cout<<"feature_in["<<h*CHin*Win+w*CHin+cin<<"]*W["<<ii*Kx*CHin*CHout+jj*CHin*CHout+cin*CHout+cout<<"]\n";
								Dtype_mul_pre1 tp,tp1;
								Dtype_mul_pre1 A=feature_in[h*CHin_Pre1*Win_Pre1+w*CHin_Pre1+cin];
//								std::cout<<"Hist:"<<A<<"\n";
								Dtype_mul_pre1 B=W[ii*Kx_Pre1*CHin_Pre1*CHout_Pre1+jj*CHin_Pre1*CHout_Pre1+cin*CHout_Pre1+cout];
//								std::cout<<"Weight:"<<B<<"\n";
//								std::cout<<"weight_index:"<<ii*Kx_Pre1*CHin_Pre1*CHout_Pre1+jj*CHin_Pre1*CHout_Pre1+cin*CHout_Pre1+cout<<"\n";
//								std::cout<<"ii:"<<ii<<",jj:"<<jj<<",cout:"<<cout<<"\n";
//								std::cout << "****" <<std::endl;
								if(A>=B)
									tp=A-B;
								else if (A<B)
									tp=B-A;
								tp1=-tp;
								sum+=tp1;
							}
						}
					}
				sum=scale[cout]*sum+shift[cout];//BN
				if (sum<0) sum=0;
				//sum=(sum>0)?sum:0;//ReLu
				feature_out[i*Wout_Pre1*CHout_Pre1+j*CHout_Pre1+cout]=sum;
			}
}
