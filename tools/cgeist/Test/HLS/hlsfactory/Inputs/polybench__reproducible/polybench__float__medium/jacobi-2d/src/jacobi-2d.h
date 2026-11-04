#pragma once
#include <cmath>


extern "C" {
void kernel_jacobi_2d(
			    
			    double A[ 250 + 0][250 + 0],
			    double B[ 250 + 0][250 + 0]);
}