#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_3mm(    
		t_ap_fixed E[ 40 + 0][50 + 0],
		t_ap_fixed A[ 40 + 0][60 + 0],
		t_ap_fixed B[ 60 + 0][50 + 0],
		t_ap_fixed F[ 50 + 0][70 + 0],
		t_ap_fixed C[ 50 + 0][80 + 0],
		t_ap_fixed D[ 80 + 0][70 + 0],
		t_ap_fixed G[ 40 + 0][70 + 0]);
}