#pragma once
#include <cmath>


extern "C" {
void kernel_mvt(
		double x1[ 400 + 0],
		double x2[ 400 + 0],
		double y_1[ 400 + 0],
		double y_2[ 400 + 0],
		double A[ 400 + 0][400 + 0]);
}