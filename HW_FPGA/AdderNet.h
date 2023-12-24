#include <ap_int.h>
#include <iostream>
#include <ap_fixed.h>
#include "Pre1.h"
#include "Pre2.h"
#include "Resblock.h"
#include "O1_1.h"
#include "O1_2.h"
#include "O1_3.h"
#include "O2_1.h"
#include "O2_2.h"
#include "O2_3.h"

void AdderNet_fixedp_1 (Dtype_f_pre1_i hist_in1[Win_Pre1*Hin_Pre1*CHin_Pre1],
		Dtype_f Lifetimes_out1[2]
);

void AdderNet_fixedp_2 (Dtype_f_pre1_i hist_in1[Win_Pre1*Hin_Pre1*CHin_Pre1],
		Dtype_f Lifetimes_out1[2]
);

void AdderNet_fixedp_3 (Dtype_f_pre1_i hist_in1[Win_Pre1*Hin_Pre1*CHin_Pre1],
		Dtype_f Lifetimes_out1[2]
);

void AdderNet_fixedp_4 (Dtype_f_pre1_i hist_in1[Win_Pre1*Hin_Pre1*CHin_Pre1],
		Dtype_f Lifetimes_out1[2]
);
