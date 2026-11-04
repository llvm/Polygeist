#pragma once
#include <cmath>


extern "C" {
void kernel_syr2k( 
		  double alpha,
		  double beta,
		  double C[ 240 + 0][240 + 0],
		  double A[ 240 + 0][200 + 0],
		  double B[ 240 + 0][200 + 0]);
}