#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_mvt(
		t_ap_fixed x1[ 40 + 0],
		t_ap_fixed x2[ 40 + 0],
		t_ap_fixed y_1[ 40 + 0],
		t_ap_fixed y_2[ 40 + 0],
		t_ap_fixed A[ 40 + 0][40 + 0]);
}