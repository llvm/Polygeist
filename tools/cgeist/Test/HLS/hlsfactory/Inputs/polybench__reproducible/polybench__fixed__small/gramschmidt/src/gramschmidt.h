#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_gramschmidt( 
			t_ap_fixed A[ 60 + 0][80 + 0],
			t_ap_fixed R[ 80 + 0][80 + 0],
			t_ap_fixed Q[ 60 + 0][80 + 0]);
}