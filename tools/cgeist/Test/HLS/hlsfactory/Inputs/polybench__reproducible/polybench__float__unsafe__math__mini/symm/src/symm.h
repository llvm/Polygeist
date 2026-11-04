#pragma once
#include <cmath>


extern "C" {
void kernel_symm( 
		 double alpha,
		 double beta,
		 double C[ 20 + 0][30 + 0],
		 double A[ 20 + 0][20 + 0],
		 double B[ 20 + 0][30 + 0]);
}