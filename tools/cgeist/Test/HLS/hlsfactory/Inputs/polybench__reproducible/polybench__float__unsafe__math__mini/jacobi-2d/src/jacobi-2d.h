#pragma once
#include <cmath>


extern "C" {
void kernel_jacobi_2d(
			    
			    double A[ 30 + 0][30 + 0],
			    double B[ 30 + 0][30 + 0]);
}