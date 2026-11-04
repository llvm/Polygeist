#pragma once
#include <cmath>


extern "C" {
void kernel_symm( 
		 double alpha,
		 double beta,
		 double C[ 200 + 0][240 + 0],
		 double A[ 200 + 0][200 + 0],
		 double B[ 200 + 0][240 + 0]);
}