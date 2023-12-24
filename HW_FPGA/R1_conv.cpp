#include "R1_conv.h"
#include "ap_int.h"
#include "Pre1.h"

static Dtype_para W[Kx_R1*Ky_R1*CHin_R1*CHout_R1]={
		#include "./parameters/R1_weight.h"
		};
static Dtype_para scale[CHout_R1]={
		#include "./parameters/R1_scale.h"
		};
static Dtype_para shift[CHout_R1]={
		#include "./parameters/R1_shift.h"
		};

void R1_conv(
		Dtype_f_r1 feature_in[Win_R1*Hin_R1*CHin_R1],
		Dtype_f_r1 feature_out[Wout_R1*Hout_R1*CHout_R1]
	)
{
//#pragma HLS array_partition variable=W complete
//#pragma HLS array_partition variable=scale complete
//#pragma HLS array_partition variable=shift complete
//#pragma HLS array_partition variable=feature_in complete
//#pragma HLS array_partition variable=feature_out complete
//#pragma HLS array_partition variable=feature_out block factor=10

	for(int cout=0;cout<CHout_R1;cout++)
		for(int i=0;i<Hout_R1;i++)
			for(int j=0;j<Wout_R1;j++)
			{
				Dtype_f_r1 sum=0;
				for(int ii=0;ii<Ky_R1;ii++)
					for(int jj=0;jj<Kx_R1;jj++)
					{
						ap_int<16> h=i*Sy_R1+ii;
						ap_int<16> w=j*Sx_R1+jj;
						if(h>=0 && w>=0 && h<Hin_R1 && w<Win_R1)
						{
							//#pragma HLS PIPELINE
							for(int cin=0;cin<CHin_R1;cin++)
							{
								#pragma HLS PIPELINE
								//Feature [H][W][C]
								//kernel: [Ky][Kx][CHin][CHout]
								//Dtype_mul tp=feature_in[h][w][cin]*w[ii][jj][cin][cout];
								//std::cout<<"h:"<<h<<",w"<<w<<",cin"<<cin<<"\n";
								//std::cout<<"feature_in["<<h*CHin*Win+w*CHin+cin<<"]*W["<<ii*Kx*CHin*CHout+jj*CHin*CHout+cin*CHout+cout<<"]\n";
								Dtype_mul_r1 tp,tp1;
								Dtype_mul_r1 A=feature_in[h*CHin_R1*Win_R1+w*CHin_R1+cin];
//								std::cout<<"Hist:"<<A<<"\n";
								Dtype_mul_r1 B=W[ii*Kx_R1*CHin_R1*CHout_R1+jj*CHin_R1*CHout_R1+cin*CHout_R1+cout];
//								std::cout<<"Weight:"<<B<<"\n";
//								std::cout<<"weight_index:"<<ii*Kx_R1*CHin_R1*CHout_R1+jj*CHin_R1*CHout_R1+cin*CHout_R1+cout<<"\n";
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
				if(sum<0)
					sum=0;
				//sum=(sum>0)?sum:0;//ReLu
				feature_out[i*Wout_R1*CHout_R1+j*CHout_R1+cout]=sum;
			}
}
