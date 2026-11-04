#pragma once
#include <cmath>


extern "C" {
void kernel_syr2k( 
		  double alpha,
		  double beta,
		  double C[ 30 + 0][30 + 0],
		  double A[ 30 + 0][20 + 0],
		  double B[ 30 + 0][20 + 0]);
}