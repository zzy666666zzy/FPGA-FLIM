#include "Pre2.h"
#include "ap_int.h"

static Dtype_para W[Kx_Pre2*Ky_Pre2*CHin_Pre2*CHout_Pre2]={
		#include "./parameters/Pre2_weight.h"
		};
static Dtype_para scale[CHout_Pre2]={
		#include "./parameters/Pre2_scale.h"
		};
static Dtype_para shift[CHout_Pre2]={
		#include "./parameters/Pre2_shift.h"
		};

void Pre2_conv(
		Dtype_f_pre2 feature_in[Win_Pre2*Hin_Pre2*CHin_Pre2],
		Dtype_f_pre2 feature_out[Wout_Pre2*Hout_Pre2*CHout_Pre2]
	)
{
	//Factor should be divisible
//	#pragma HLS ARRAY_PARTITION variable=W complete
//	#pragma HLS ARRAY_PARTITION variable=scale block factor=10
//	#pragma HLS ARRAY_PARTITION variable=shift block factor=10
//	#pragma HLS ARRAY_PARTITION variable=feature_in complete
	//#pragma HLS ARRAY_PARTITION variable=feature_out block factor=10

	for(int cout=0;cout<CHout_Pre2;cout++)
		for(int i=0;i<Hout_Pre2;i++)
			for(int j=0;j<Wout_Pre2;j++)
			{
				Dtype_acc_pre2 sum=0;
				for(int ii=0;ii<Ky_Pre2;ii++)
					for(int jj=0;jj<Kx_Pre2;jj++)
					{
						ap_int<16> h=i*Sy_Pre2+ii;
						ap_int<16> w=j*Sx_Pre2+jj;
						if(h>=0 && w>=0 && h<Hin_Pre2 && w<Win_Pre2)
						{
							#pragma HLS PIPELINE
							for(int cin=0;cin<CHin_Pre2;cin++)
							{
								//Feature [H][W][C]
								//kernel: [Ky][Kx][CHin][CHout]
								//Dtype_mul tp=feature_in[h][w][cin]*w[ii][jj][cin][cout];
								//std::cout<<"h:"<<h<<",w"<<w<<",cin"<<cin<<"\n";
								//std::cout<<"feature_in["<<h*CHin*Win+w*CHin+cin<<"]*W["<<ii*Kx*CHin*CHout+jj*CHin*CHout+cin*CHout+cout<<"]\n";
								Dtype_mul_pre2 tp,tp1;
								Dtype_mul_pre2 A=feature_in[h*CHin_Pre2*Win_Pre2+w*CHin_Pre2+cin];
//								std::cout<<"Hist:"<<A<<"\n";
								Dtype_mul_pre2 B=W[ii*Kx_Pre2*CHin_Pre2*CHout_Pre2+jj*CHin_Pre2*CHout_Pre2+cin*CHout_Pre2+cout];
//								std::cout<<"Weight:"<<B<<"\n";
//								std::cout<<"weight_index:"<<ii*Kx_Pre2*CHin_Pre2*CHout_Pre2+jj*CHin_Pre2*CHout_Pre2+cin*CHout_Pre2+cout<<"\n";
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
				feature_out[i*Wout_Pre2*CHout_Pre2+j*CHout_Pre2+cout]=sum;
			}
}
