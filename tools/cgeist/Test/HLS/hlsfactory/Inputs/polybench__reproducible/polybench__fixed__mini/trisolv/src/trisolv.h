#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_trisolv(
		    t_ap_fixed L[ 40 + 0][40 + 0],
		    t_ap_fixed x[ 40 + 0],
		    t_ap_fixed b[ 40 + 0]);
}