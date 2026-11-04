#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_bicg( 
		 t_ap_fixed A[ 42 + 0][38 + 0],
		 t_ap_fixed s[ 38 + 0],
		 t_ap_fixed q[ 42 + 0],
		 t_ap_fixed p[ 38 + 0],
		 t_ap_fixed r[ 42 + 0]);
}