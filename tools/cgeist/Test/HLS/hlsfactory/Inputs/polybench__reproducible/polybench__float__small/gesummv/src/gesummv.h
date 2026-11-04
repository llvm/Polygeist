#pragma once
#include <cmath>


extern "C" {
void kernel_gesummv(
		    double alpha,
		    double beta,
		    double A[ 90 + 0][90 + 0],
		    double B[ 90 + 0][90 + 0],
		    double tmp[ 90 + 0],
		    double x[ 90 + 0],
		    double y[ 90 + 0]);
}