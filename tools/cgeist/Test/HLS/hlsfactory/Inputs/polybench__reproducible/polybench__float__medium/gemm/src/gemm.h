#pragma once
#include <cmath>


extern "C" {
void kernel_gemm(  
		 double alpha,
		 double beta,
		 double C[ 200 + 0][220 + 0],
		 double A[ 200 + 0][240 + 0],
		 double B[ 240 + 0][220 + 0]);
}