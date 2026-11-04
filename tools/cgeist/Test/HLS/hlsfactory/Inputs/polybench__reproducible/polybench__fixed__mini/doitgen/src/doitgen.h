#pragma once
#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32,16> t_ap_fixed;

extern "C" {
void kernel_doitgen(  
		    t_ap_fixed A[ 10 + 0][8 + 0][12 + 0],
		    t_ap_fixed C4[ 12 + 0][12 + 0],
		    t_ap_fixed sum[ 12 + 0]);
}