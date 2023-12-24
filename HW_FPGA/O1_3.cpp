#include "Pre1.h"
#include "O1_3.h"
#include "AdderNet.h"

static Dtype_f W[Kx_O1_3*Ky_O1_3*CHin_O1_3*CHout_O1_3]={
		#include "./parameters/O1_3_weight.h"
		};
static Dtype_f scale[CHout_O1_3]={
		#include "./parameters/O1_3_scale.h"
		};
static Dtype_f shift[CHout_O1_3]={
		#include "./parameters/O1_3_shift.h"
		};

void O1_3(
		Dtype_f feature_in[Win_O1_3*Hin_O1_3*CHin_O1_3],
		Dtype_f feature_out[Wout_O1_3*Hout_O1_3*CHout_O1_3]
	)
{

	for(int cout=0;cout<CHout_O1_3;cout++)
		for(int i=0;i<Hout_O1_3;i++)
			for(int j=0;j<Wout_O1_3;j++)
			{
				Dtype_acc sum=0;
				for(int ii=0;ii<Ky_O1_3;ii++)
					for(int jj=0;jj<Kx_O1_3;jj++)
					{
						ap_int<16> h=i*Sy_O1_3+ii;
						ap_int<16> w=j*Sx_O1_3+jj;
						if(h>=0 && w>=0 && h<Hin_O1_3 && w<Win_O1_3)
						{
							for(int cin=0;cin<CHin_O1_3;cin++)
							{
								#pragma HLS PIPELINE
								//Feature [H][W][C]
								//kernel: [Ky][Kx][CHin][CHout]
								//Dtype_mul tp=feature_in[h][w][cin]*w[ii][jj][cin][cout];
								//std::cout<<"h:"<<h<<",w"<<w<<",cin"<<cin<<"\n";
								//std::cout<<"feature_in["<<h*CHin*Win+w*CHin+cin<<"]*W["<<ii*Kx*CHin*CHout+jj*CHin*CHout+cin*CHout+cout<<"]\n";
								Dtype_mul tp,tp1;
								Dtype_mul A=feature_in[h*CHin_O1_3*Win_O1_3+w*CHin_O1_3+cin];
//								std::cout<<"Hist:"<<A<<"\n";
								Dtype_mul B=W[ii*Kx_O1_3*CHin_O1_3*CHout_O1_3+jj*CHin_O1_3*CHout_O1_3+cin*CHout_O1_3+cout];
//								std::cout<<"Weight:"<<B<<"\n";
//								std::cout<<"weight_index:"<<ii*Kx_O1_3*CHin_O1_3*CHout_O1_3+jj*CHin_O1_3*CHout_O1_3+cin*CHout_O1_3+cout<<"\n";
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
				feature_out[i*Wout_O1_3*CHout_O1_3+j*CHout_O1_3+cout]=sum;
			}
}
