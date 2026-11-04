#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_syr2k( 
		  t_ap_fixed alpha,
		  t_ap_fixed beta,
		  t_ap_fixed C[ 30 + 0][30 + 0],
		  t_ap_fixed A[ 30 + 0][20 + 0],
		  t_ap_fixed B[ 30 + 0][20 + 0]);
}