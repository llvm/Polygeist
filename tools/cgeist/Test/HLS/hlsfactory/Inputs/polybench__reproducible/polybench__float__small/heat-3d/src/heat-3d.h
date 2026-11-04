#pragma once
#include <cmath>


extern "C" {
void kernel_heat_3d(
		      
		      double A[ 20 + 0][20 + 0][20 + 0],
		      double B[ 20 + 0][20 + 0][20 + 0]);
}