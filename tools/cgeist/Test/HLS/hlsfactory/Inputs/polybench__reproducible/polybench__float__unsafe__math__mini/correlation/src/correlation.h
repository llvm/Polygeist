#pragma once
#include <cmath>


extern "C" {
void kernel_correlation( 
			double float_n,
			double data[ 32 + 0][28 + 0],
			double corr[ 28 + 0][28 + 0],
			double mean[ 28 + 0],
			double stddev[ 28 + 0]);
}