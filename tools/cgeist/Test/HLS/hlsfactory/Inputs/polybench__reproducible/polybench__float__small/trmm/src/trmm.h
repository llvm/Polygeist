#pragma once
#include <cmath>


extern "C" {
void kernel_trmm( 
		 double alpha,
		 double A[ 60 + 0][60 + 0],
		 double B[ 60 + 0][80 + 0]);
}