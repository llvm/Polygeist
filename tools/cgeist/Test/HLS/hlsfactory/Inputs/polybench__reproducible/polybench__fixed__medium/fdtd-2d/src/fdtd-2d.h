#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_fdtd_2d(
		    
		    
		    t_ap_fixed ex[ 200 + 0][240 + 0],
		    t_ap_fixed ey[ 200 + 0][240 + 0],
		    t_ap_fixed hz[ 200 + 0][240 + 0],
		    t_ap_fixed _fict_[ 100 + 0]);
}