#pragma once
#include <cmath>


extern "C" {
void kernel_mvt(
		double x1[ 40 + 0],
		double x2[ 40 + 0],
		double y_1[ 40 + 0],
		double y_2[ 40 + 0],
		double A[ 40 + 0][40 + 0]);
}