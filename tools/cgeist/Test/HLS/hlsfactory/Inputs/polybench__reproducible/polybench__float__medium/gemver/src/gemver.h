#pragma once
#include <cmath>


extern "C" {
void kernel_gemver(
		   double alpha,
		   double beta,
		   double A[ 400 + 0][400 + 0],
		   double u1[ 400 + 0],
		   double v1[ 400 + 0],
		   double u2[ 400 + 0],
		   double v2[ 400 + 0],
		   double w[ 400 + 0],
		   double x[ 400 + 0],
		   double y[ 400 + 0],
		   double z[ 400 + 0]);
}