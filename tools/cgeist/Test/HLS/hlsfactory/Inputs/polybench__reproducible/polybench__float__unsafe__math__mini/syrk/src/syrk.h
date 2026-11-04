#pragma once
#include <cmath>


extern "C" {
void kernel_syrk( 
		 double alpha,
		 double beta,
		 double C[ 30 + 0][30 + 0],
		 double A[ 30 + 0][20 + 0]);
}