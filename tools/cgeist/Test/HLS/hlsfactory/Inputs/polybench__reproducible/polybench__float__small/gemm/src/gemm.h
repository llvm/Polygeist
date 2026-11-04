#pragma once
#include <cmath>


extern "C" {
void kernel_gemm(  
		 double alpha,
		 double beta,
		 double C[ 60 + 0][70 + 0],
		 double A[ 60 + 0][80 + 0],
		 double B[ 80 + 0][70 + 0]);
}