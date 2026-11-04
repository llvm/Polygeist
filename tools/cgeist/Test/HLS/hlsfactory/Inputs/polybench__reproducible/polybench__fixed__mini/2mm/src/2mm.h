#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_2mm(   
		t_ap_fixed alpha,
		t_ap_fixed beta,
		t_ap_fixed tmp[ 16 + 0][18 + 0],
		t_ap_fixed A[ 16 + 0][22 + 0],
		t_ap_fixed B[ 22 + 0][18 + 0],
		t_ap_fixed C[ 18 + 0][24 + 0],
		t_ap_fixed D[ 16 + 0][24 + 0]);
}