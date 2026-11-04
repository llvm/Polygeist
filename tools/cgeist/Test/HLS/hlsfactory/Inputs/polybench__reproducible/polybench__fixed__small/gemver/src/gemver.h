#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_gemver(
		   t_ap_fixed alpha,
		   t_ap_fixed beta,
		   t_ap_fixed A[ 120 + 0][120 + 0],
		   t_ap_fixed u1[ 120 + 0],
		   t_ap_fixed v1[ 120 + 0],
		   t_ap_fixed u2[ 120 + 0],
		   t_ap_fixed v2[ 120 + 0],
		   t_ap_fixed w[ 120 + 0],
		   t_ap_fixed x[ 120 + 0],
		   t_ap_fixed y[ 120 + 0],
		   t_ap_fixed z[ 120 + 0]);
}