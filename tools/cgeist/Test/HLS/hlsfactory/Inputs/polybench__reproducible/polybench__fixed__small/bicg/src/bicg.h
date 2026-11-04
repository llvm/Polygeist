#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_bicg( 
		 t_ap_fixed A[ 124 + 0][116 + 0],
		 t_ap_fixed s[ 116 + 0],
		 t_ap_fixed q[ 124 + 0],
		 t_ap_fixed p[ 116 + 0],
		 t_ap_fixed r[ 124 + 0]);
}