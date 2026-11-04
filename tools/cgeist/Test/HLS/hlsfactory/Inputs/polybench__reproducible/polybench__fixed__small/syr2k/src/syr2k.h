#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_syr2k( 
		  t_ap_fixed alpha,
		  t_ap_fixed beta,
		  t_ap_fixed C[ 80 + 0][80 + 0],
		  t_ap_fixed A[ 80 + 0][60 + 0],
		  t_ap_fixed B[ 80 + 0][60 + 0]);
}