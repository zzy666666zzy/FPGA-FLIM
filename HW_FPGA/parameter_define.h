#include <ap_int.h>
#include <iostream>
#include <ap_fixed.h>
//-------------------Pre1-------------------
#define CHin_Pre1 1
#define Hin_Pre1 1
#define	Win_Pre1 256
#define CHout_Pre1 5
#define Kx_Pre1 13
#define Ky_Pre1 1
#define Sx_Pre1 5
#define	Sy_Pre1 1
#define Wout_Pre1 49
#define Hout_Pre1 1

typedef ap_fixed<18,6> Dtype_para;//make sure fractional part has enough bits.

typedef ap_fixed<32,16> Dtype_f;
typedef ap_fixed<32,16> Dtype_acc;
typedef ap_fixed<32,16> Dtype_mul;

typedef ap_fixed<32,16> Dtype_f_pre1_i;
typedef ap_fixed<32,16> Dtype_mul_pre1;
typedef ap_fixed<32,16> Dtype_f_pre1_o;

//---------------Pre2--------------------
#define CHin_Pre2 5
#define Hin_Pre2 1
#define	Win_Pre2 49
#define CHout_Pre2 10
#define Kx_Pre2 9
#define Ky_Pre2 1
#define Sx_Pre2 3
#define	Sy_Pre2 1
#define Wout_Pre2 14
#define Hout_Pre2 1

typedef ap_fixed<32,16> Dtype_f_pre2;
typedef ap_fixed<32,16> Dtype_acc_pre2;
typedef ap_fixed<32,16> Dtype_mul_pre2;

//-------------R1-----------------
#define CHin_R1 10
#define Hin_R1 1
#define	Win_R1 14
#define CHout_R1 10
#define Kx_R1 1
#define Ky_R1 1
#define Sx_R1 1
#define	Sy_R1 1
#define Wout_R1 14
#define Hout_R1 1

typedef ap_fixed<32,16> Dtype_f_r1;
typedef ap_fixed<32,16> Dtype_acc_r1;
typedef ap_fixed<32,16> Dtype_mul_r1;

//--------------R2------------------
#define CHin_R2 10
#define Hin_R2 1
#define	Win_R2 14
#define CHout_R2 10
#define Kx_R2 1
#define Ky_R2 1
#define Sx_R2 1
#define	Sy_R2 1
#define Wout_R2 14
#define Hout_R2 1

typedef ap_fixed<32,16> Dtype_f_r2;
typedef ap_fixed<32,16> Dtype_acc_r2;
typedef ap_fixed<32,16> Dtype_mul_r2;

//----------------O1_1-------------------
#define CHin_O1_1 140
#define Hin_O1_1 1
#define	Win_O1_1 1
#define CHout_O1_1 70
#define Kx_O1_1 1
#define Ky_O1_1 1
#define Sx_O1_1 1
#define	Sy_O1_1 1
#define Wout_O1_1 1
#define Hout_O1_1 1

//----------------O1_2-------------------
#define CHin_O1_2 70
#define Hin_O1_2 1
#define	Win_O1_2 1
#define CHout_O1_2 30
#define Kx_O1_2 1
#define Ky_O1_2 1
#define Sx_O1_2 1
#define	Sy_O1_2 1
#define Wout_O1_2 1
#define Hout_O1_2 1

//----------------O1_3-------------------
#define CHin_O1_3 30
#define Hin_O1_3 1
#define	Win_O1_3 1
#define CHout_O1_3 1
#define Kx_O1_3 1
#define Ky_O1_3 1
#define Sx_O1_3 1
#define	Sy_O1_3 1
#define Wout_O1_3 1
#define Hout_O1_3 1

//----------------O2_1-------------------
#define CHin_O2_1 140
#define Hin_O2_1 1
#define	Win_O2_1 1
#define CHout_O2_1 70
#define Kx_O2_1 1
#define Ky_O2_1 1
#define Sx_O2_1 1
#define	Sy_O2_1 1
#define Wout_O2_1 1
#define Hout_O2_1 1

//----------------O2_2-------------------
#define CHin_O2_2 70
#define Hin_O2_2 1
#define	Win_O2_2 1
#define CHout_O2_2 30
#define Kx_O2_2 1
#define Ky_O2_2 1
#define Sx_O2_2 1
#define	Sy_O2_2 1
#define Wout_O2_2 1
#define Hout_O2_2 1

//----------------O2_3-------------------
#define CHin_O2_3 30
#define Hin_O2_3 1
#define	Win_O2_3 1
#define CHout_O2_3 1
#define Kx_O2_3 1
#define Ky_O2_3 1
#define Sx_O2_3 1
#define	Sy_O2_3 1
#define Wout_O2_3 1
#define Hout_O2_3 1
