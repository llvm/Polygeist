#pragma once
#include <cmath>


extern "C" {
void kernel_ludcmp(
		   double A[ 120 + 0][120 + 0],
		   double b[ 120 + 0],
		   double x[ 120 + 0],
		   double y[ 120 + 0]);
}