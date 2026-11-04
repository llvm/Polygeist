#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_gemm(  
		 t_ap_fixed alpha,
		 t_ap_fixed beta,
		 t_ap_fixed C[ 60 + 0][70 + 0],
		 t_ap_fixed A[ 60 + 0][80 + 0],
		 t_ap_fixed B[ 80 + 0][70 + 0]);
}