#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_fdtd_2d(
		    
		    
		    t_ap_fixed ex[ 20 + 0][30 + 0],
		    t_ap_fixed ey[ 20 + 0][30 + 0],
		    t_ap_fixed hz[ 20 + 0][30 + 0],
		    t_ap_fixed _fict_[ 20 + 0]);
}