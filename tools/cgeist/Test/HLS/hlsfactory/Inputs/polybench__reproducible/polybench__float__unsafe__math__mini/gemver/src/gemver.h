#pragma once
#include <cmath>


extern "C" {
void kernel_gemver(
		   double alpha,
		   double beta,
		   double A[ 40 + 0][40 + 0],
		   double u1[ 40 + 0],
		   double v1[ 40 + 0],
		   double u2[ 40 + 0],
		   double v2[ 40 + 0],
		   double w[ 40 + 0],
		   double x[ 40 + 0],
		   double y[ 40 + 0],
		   double z[ 40 + 0]);
}