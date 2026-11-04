#pragma once
#include <cmath>


extern "C" {
void kernel_covariance( 
		       double float_n,
		       double data[ 260 + 0][240 + 0],
		       double cov[ 240 + 0][240 + 0],
		       double mean[ 240 + 0]);
}