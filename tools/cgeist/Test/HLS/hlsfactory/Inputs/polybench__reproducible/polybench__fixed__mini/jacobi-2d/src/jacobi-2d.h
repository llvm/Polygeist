#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_jacobi_2d(
			    
			    t_ap_fixed A[ 30 + 0][30 + 0],
			    t_ap_fixed B[ 30 + 0][30 + 0]);
}