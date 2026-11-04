#pragma once
#include <cmath>


extern "C" {
void kernel_jacobi_2d(
			    
			    double A[ 90 + 0][90 + 0],
			    double B[ 90 + 0][90 + 0]);
}