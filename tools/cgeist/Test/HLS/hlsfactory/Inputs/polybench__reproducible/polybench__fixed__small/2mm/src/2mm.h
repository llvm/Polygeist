#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_2mm(   
		t_ap_fixed alpha,
		t_ap_fixed beta,
		t_ap_fixed tmp[ 40 + 0][50 + 0],
		t_ap_fixed A[ 40 + 0][70 + 0],
		t_ap_fixed B[ 70 + 0][50 + 0],
		t_ap_fixed C[ 50 + 0][80 + 0],
		t_ap_fixed D[ 40 + 0][80 + 0]);
}