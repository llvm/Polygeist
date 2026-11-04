#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_gesummv(
		    t_ap_fixed alpha,
		    t_ap_fixed beta,
		    t_ap_fixed A[ 30 + 0][30 + 0],
		    t_ap_fixed B[ 30 + 0][30 + 0],
		    t_ap_fixed tmp[ 30 + 0],
		    t_ap_fixed x[ 30 + 0],
		    t_ap_fixed y[ 30 + 0]);
}