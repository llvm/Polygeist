#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_3mm(    
		t_ap_fixed E[ 16 + 0][18 + 0],
		t_ap_fixed A[ 16 + 0][20 + 0],
		t_ap_fixed B[ 20 + 0][18 + 0],
		t_ap_fixed F[ 18 + 0][22 + 0],
		t_ap_fixed C[ 18 + 0][24 + 0],
		t_ap_fixed D[ 24 + 0][22 + 0],
		t_ap_fixed G[ 16 + 0][22 + 0]);
}