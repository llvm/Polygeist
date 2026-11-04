#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_heat_3d(
		      
		      t_ap_fixed A[ 10 + 0][10 + 0][10 + 0],
		      t_ap_fixed B[ 10 + 0][10 + 0][10 + 0]);
}