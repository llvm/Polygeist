#pragma once
#include <cmath>


extern "C" {
void kernel_syrk( 
		 double alpha,
		 double beta,
		 double C[ 80 + 0][80 + 0],
		 double A[ 80 + 0][60 + 0]);
}