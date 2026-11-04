#pragma once
#include <cmath>


extern "C" {
void kernel_heat_3d(
		      
		      double A[ 40 + 0][40 + 0][40 + 0],
		      double B[ 40 + 0][40 + 0][40 + 0]);
}