#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_bicg( 
		 t_ap_fixed A[ 410 + 0][390 + 0],
		 t_ap_fixed s[ 390 + 0],
		 t_ap_fixed q[ 410 + 0],
		 t_ap_fixed p[ 390 + 0],
		 t_ap_fixed r[ 410 + 0]);
}