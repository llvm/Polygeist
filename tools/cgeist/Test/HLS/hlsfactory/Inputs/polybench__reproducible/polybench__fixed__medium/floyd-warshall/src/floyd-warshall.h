#pragma once
#include "ap_fixed.h"
#include "hls_math.h"


extern "C" {
void kernel_floyd_warshall(
			   int path[ 500 + 0][500 + 0]);
}