#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_covariance( 
		       t_ap_fixed float_n,
		       t_ap_fixed data[ 260 + 0][240 + 0],
		       t_ap_fixed cov[ 240 + 0][240 + 0],
		       t_ap_fixed mean[ 240 + 0]);
}