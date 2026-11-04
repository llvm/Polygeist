#pragma once
#include <cmath>


extern "C" {
void kernel_mvt(
		double x1[ 120 + 0],
		double x2[ 120 + 0],
		double y_1[ 120 + 0],
		double y_2[ 120 + 0],
		double A[ 120 + 0][120 + 0]);
}