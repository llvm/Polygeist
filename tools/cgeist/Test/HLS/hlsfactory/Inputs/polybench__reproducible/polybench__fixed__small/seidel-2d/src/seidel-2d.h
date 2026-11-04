#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_seidel_2d(
		      
		      t_ap_fixed A[ 120 + 0][120 + 0]);
}