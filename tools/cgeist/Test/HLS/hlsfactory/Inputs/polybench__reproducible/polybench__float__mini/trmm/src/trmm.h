#pragma once
#include <cmath>


extern "C" {
void kernel_trmm( 
		 double alpha,
		 double A[ 20 + 0][20 + 0],
		 double B[ 20 + 0][30 + 0]);
}