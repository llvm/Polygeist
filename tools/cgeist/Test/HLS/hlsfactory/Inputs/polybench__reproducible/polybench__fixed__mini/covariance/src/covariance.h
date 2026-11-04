#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_covariance( 
		       t_ap_fixed float_n,
		       t_ap_fixed data[ 32 + 0][28 + 0],
		       t_ap_fixed cov[ 28 + 0][28 + 0],
		       t_ap_fixed mean[ 28 + 0]);
}