#pragma once
#include "ap_fixed.h"
#include "hls_math.h"


extern "C" {
void kernel_nussinov( char seq[ 180 + 0],
			   int table[ 180 + 0][180 + 0]);
}