#pragma once
#include <cmath>


extern "C" {
void kernel_heat_3d(
		      
		      double A[ 10 + 0][10 + 0][10 + 0],
		      double B[ 10 + 0][10 + 0][10 + 0]);
}