#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_doitgen(  
		    t_ap_fixed A[ 50 + 0][40 + 0][60 + 0],
		    t_ap_fixed C4[ 60 + 0][60 + 0],
		    t_ap_fixed sum[ 60 + 0]);
}