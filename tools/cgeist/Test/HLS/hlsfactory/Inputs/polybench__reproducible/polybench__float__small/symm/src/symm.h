#pragma once
#include <cmath>


extern "C" {
void kernel_symm( 
		 double alpha,
		 double beta,
		 double C[ 60 + 0][80 + 0],
		 double A[ 60 + 0][60 + 0],
		 double B[ 60 + 0][80 + 0]);
}