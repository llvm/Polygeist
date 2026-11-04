#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_doitgen(  
		    t_ap_fixed A[ 25 + 0][20 + 0][30 + 0],
		    t_ap_fixed C4[ 30 + 0][30 + 0],
		    t_ap_fixed sum[ 30 + 0]);
}