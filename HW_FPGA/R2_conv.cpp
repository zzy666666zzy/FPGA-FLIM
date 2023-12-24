#include "R2_conv.h"
#include "ap_int.h"
#include "Pre1.h"

static Dtype_para W[Kx_R2*Ky_R2*CHin_R2*CHout_R2]={
		#include "./parameters/R2_weight.h"
		};
static Dtype_para scale[CHout_R2]={
		#include "./parameters/R2_scale.h"
		};
static Dtype_para shift[CHout_R2]={
		#include "./parameters/R2_shift.h"
		};

void R2_conv(
		Dtype_f_r2 feature_in[Win_R2*Hin_R2*CHin_R2],
		Dtype_f_r2 feature_out[Wout_R2*Hout_R2*CHout_R2]
	)
{
	#pragma HLS array_partition variable=W block factor=10
	#pragma HLS array_partition variable=scale block factor=10
	#pragma HLS array_partition variable=shift block factor=10
	#pragma HLS array_partition variable=feature_in block factor=10

	for(int cout=0;cout<CHout_R2;cout++)
		for(int i=0;i<Hout_R2;i++)
			for(int j=0;j<Wout_R2;j++)
			{
				Dtype_acc_r2 sum=0;
				for(int ii=0;ii<Ky_R2;ii++)
					for(int jj=0;jj<Kx_R2;jj++)
					{
						ap_int<16> h=i*Sy_R2+ii;
						ap_int<16> w=j*Sx_R2+jj;
						if(h>=0 && w>=0 && h<Hin_R2 && w<Win_R2)
						{
							for(int cin=0;cin<CHin_R2;cin++)
							{
								#pragma HLS PIPELINE II=1
								//Feature [H][W][C]
								//kernel: [Ky][Kx][CHin][CHout]
								//Dtype_mul tp=feature_in[h][w][cin]*w[ii][jj][cin][cout];
								//std::cout<<"h:"<<h<<",w"<<w<<",cin"<<cin<<"\n";
								//std::cout<<"feature_in["<<h*CHin*Win+w*CHin+cin<<"]*W["<<ii*Kx*CHin*CHout+jj*CHin*CHout+cin*CHout+cout<<"]\n";
								Dtype_mul_r2 tp,tp1;
								Dtype_mul_r2 A=feature_in[h*CHin_R2*Win_R2+w*CHin_R2+cin];
//								std::cout<<"Hist:"<<A<<"\n";
								Dtype_mul_r2 B=W[ii*Kx_R2*CHin_R2*CHout_R2+jj*CHin_R2*CHout_R2+cin*CHout_R2+cout];
//								std::cout<<"Weight:"<<B<<"\n";
//								std::cout<<"weight_index:"<<ii*Kx_R2*CHin_R2*CHout_R2+jj*CHin_R2*CHout_R2+cin*CHout_R2+cout<<"\n";
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
				feature_out[i*Wout_R2*CHout_R2+j*CHout_R2+cout]=sum;
			}
}
