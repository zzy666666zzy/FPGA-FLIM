#include "Pre1.h"
#include "O2_2.h"
#include "AdderNet.h"

static Dtype_f W[Kx_O2_2*Ky_O2_2*CHin_O2_2*CHout_O2_2]={
		#include "./parameters/O2_2_weight.h"
		};
static Dtype_f scale[CHout_O2_2]={
		#include "./parameters/O2_2_scale.h"
		};
static Dtype_f shift[CHout_O2_2]={
		#include "./parameters/O2_2_shift.h"
		};

void O2_2(
		Dtype_f feature_in[Win_O2_2*Hin_O2_2*CHin_O2_2],
		Dtype_f feature_out[Wout_O2_2*Hout_O2_2*CHout_O2_2]
	)
{

	for(int cout=0;cout<CHout_O2_2;cout++)
		for(int i=0;i<Hout_O2_2;i++)
			for(int j=0;j<Wout_O2_2;j++)
			{
				Dtype_acc sum=0;
				for(int ii=0;ii<Ky_O2_2;ii++)
					for(int jj=0;jj<Kx_O2_2;jj++)
					{
						ap_int<16> h=i*Sy_O2_2+ii;
						ap_int<16> w=j*Sx_O2_2+jj;
						if(h>=0 && w>=0 && h<Hin_O2_2 && w<Win_O2_2)
						{
							for(int cin=0;cin<CHin_O2_2;cin++)
							{
								#pragma HLS PIPELINE
								//Feature [H][W][C]
								//kernel: [Ky][Kx][CHin][CHout]
								//Dtype_mul tp=feature_in[h][w][cin]*w[ii][jj][cin][cout];
								//std::cout<<"h:"<<h<<",w"<<w<<",cin"<<cin<<"\n";
								//std::cout<<"feature_in["<<h*CHin*Win+w*CHin+cin<<"]*W["<<ii*Kx*CHin*CHout+jj*CHin*CHout+cin*CHout+cout<<"]\n";
								Dtype_mul tp,tp1;
								Dtype_mul A=feature_in[h*CHin_O2_2*Win_O2_2+w*CHin_O2_2+cin];
//								std::cout<<"Hist:"<<A<<"\n";
								Dtype_mul B=W[ii*Kx_O2_2*CHin_O2_2*CHout_O2_2+jj*CHin_O2_2*CHout_O2_2+cin*CHout_O2_2+cout];
//								std::cout<<"Weight:"<<B<<"\n";
//								std::cout<<"weight_index:"<<ii*Kx_O2_2*CHin_O2_2*CHout_O2_2+jj*CHin_O2_2*CHout_O2_2+cin*CHout_O2_2+cout<<"\n";
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
				feature_out[i*Wout_O2_2*CHout_O2_2+j*CHout_O2_2+cout]=sum;
			}
}
