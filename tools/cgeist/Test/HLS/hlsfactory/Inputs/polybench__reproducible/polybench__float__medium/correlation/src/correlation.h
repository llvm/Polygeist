#pragma once
#include <cmath>


extern "C" {
void kernel_correlation( 
			double float_n,
			double data[ 260 + 0][240 + 0],
			double corr[ 240 + 0][240 + 0],
			double mean[ 240 + 0],
			double stddev[ 240 + 0]);
}