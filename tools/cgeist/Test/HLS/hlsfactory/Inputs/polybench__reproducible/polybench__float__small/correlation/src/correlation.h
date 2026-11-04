#pragma once
#include <cmath>


extern "C" {
void kernel_correlation( 
			double float_n,
			double data[ 100 + 0][80 + 0],
			double corr[ 80 + 0][80 + 0],
			double mean[ 80 + 0],
			double stddev[ 80 + 0]);
}