#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_syr2k( 
		  t_ap_fixed alpha,
		  t_ap_fixed beta,
		  t_ap_fixed C[ 240 + 0][240 + 0],
		  t_ap_fixed A[ 240 + 0][200 + 0],
		  t_ap_fixed B[ 240 + 0][200 + 0]);
}