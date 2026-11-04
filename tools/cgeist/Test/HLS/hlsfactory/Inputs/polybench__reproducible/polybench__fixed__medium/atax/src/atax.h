#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_atax( 
		 t_ap_fixed A[ 390 + 0][410 + 0],
		 t_ap_fixed x[ 410 + 0],
		 t_ap_fixed y[ 410 + 0],
		 t_ap_fixed tmp[ 390 + 0]);
}