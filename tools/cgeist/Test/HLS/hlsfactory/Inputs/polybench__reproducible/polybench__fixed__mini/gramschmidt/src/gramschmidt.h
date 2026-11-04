#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_gramschmidt( 
			t_ap_fixed A[ 20 + 0][30 + 0],
			t_ap_fixed R[ 30 + 0][30 + 0],
			t_ap_fixed Q[ 20 + 0][30 + 0]);
}