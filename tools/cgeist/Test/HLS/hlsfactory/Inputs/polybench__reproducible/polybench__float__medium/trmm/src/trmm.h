#pragma once
#include <cmath>


extern "C" {
void kernel_trmm( 
		 double alpha,
		 double A[ 200 + 0][200 + 0],
		 double B[ 200 + 0][240 + 0]);
}