#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_trisolv(
		    t_ap_fixed L[ 400 + 0][400 + 0],
		    t_ap_fixed x[ 400 + 0],
		    t_ap_fixed b[ 400 + 0]);
}