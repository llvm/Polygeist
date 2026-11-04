#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_atax( 
		 t_ap_fixed A[ 38 + 0][42 + 0],
		 t_ap_fixed x[ 42 + 0],
		 t_ap_fixed y[ 42 + 0],
		 t_ap_fixed tmp[ 38 + 0]);
}