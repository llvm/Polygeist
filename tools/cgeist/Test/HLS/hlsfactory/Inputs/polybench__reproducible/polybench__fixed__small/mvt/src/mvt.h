#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_mvt(
		t_ap_fixed x1[ 120 + 0],
		t_ap_fixed x2[ 120 + 0],
		t_ap_fixed y_1[ 120 + 0],
		t_ap_fixed y_2[ 120 + 0],
		t_ap_fixed A[ 120 + 0][120 + 0]);
}