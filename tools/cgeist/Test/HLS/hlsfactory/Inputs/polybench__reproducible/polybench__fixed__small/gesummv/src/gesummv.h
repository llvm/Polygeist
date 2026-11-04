#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_gesummv(
		    t_ap_fixed alpha,
		    t_ap_fixed beta,
		    t_ap_fixed A[ 90 + 0][90 + 0],
		    t_ap_fixed B[ 90 + 0][90 + 0],
		    t_ap_fixed tmp[ 90 + 0],
		    t_ap_fixed x[ 90 + 0],
		    t_ap_fixed y[ 90 + 0]);
}