#pragma once
#include <cmath>


extern "C" {
void kernel_gemver(
		   double alpha,
		   double beta,
		   double A[ 120 + 0][120 + 0],
		   double u1[ 120 + 0],
		   double v1[ 120 + 0],
		   double u2[ 120 + 0],
		   double v2[ 120 + 0],
		   double w[ 120 + 0],
		   double x[ 120 + 0],
		   double y[ 120 + 0],
		   double z[ 120 + 0]);
}