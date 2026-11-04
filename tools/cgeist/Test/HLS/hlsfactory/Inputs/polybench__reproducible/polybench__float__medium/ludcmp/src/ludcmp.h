#pragma once
#include <cmath>


extern "C" {
void kernel_ludcmp(
		   double A[ 400 + 0][400 + 0],
		   double b[ 400 + 0],
		   double x[ 400 + 0],
		   double y[ 400 + 0]);
}