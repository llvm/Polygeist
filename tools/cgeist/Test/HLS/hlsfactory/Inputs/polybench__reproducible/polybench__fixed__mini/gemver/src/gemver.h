#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_gemver(
		   t_ap_fixed alpha,
		   t_ap_fixed beta,
		   t_ap_fixed A[ 40 + 0][40 + 0],
		   t_ap_fixed u1[ 40 + 0],
		   t_ap_fixed v1[ 40 + 0],
		   t_ap_fixed u2[ 40 + 0],
		   t_ap_fixed v2[ 40 + 0],
		   t_ap_fixed w[ 40 + 0],
		   t_ap_fixed x[ 40 + 0],
		   t_ap_fixed y[ 40 + 0],
		   t_ap_fixed z[ 40 + 0]);
}