#pragma once
#include <cmath>


extern "C" {
void kernel_covariance( 
		       double float_n,
		       double data[ 100 + 0][80 + 0],
		       double cov[ 80 + 0][80 + 0],
		       double mean[ 80 + 0]);
}