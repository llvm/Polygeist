#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_gemver(
		   t_ap_fixed alpha,
		   t_ap_fixed beta,
		   t_ap_fixed A[ 400 + 0][400 + 0],
		   t_ap_fixed u1[ 400 + 0],
		   t_ap_fixed v1[ 400 + 0],
		   t_ap_fixed u2[ 400 + 0],
		   t_ap_fixed v2[ 400 + 0],
		   t_ap_fixed w[ 400 + 0],
		   t_ap_fixed x[ 400 + 0],
		   t_ap_fixed y[ 400 + 0],
		   t_ap_fixed z[ 400 + 0]);
}