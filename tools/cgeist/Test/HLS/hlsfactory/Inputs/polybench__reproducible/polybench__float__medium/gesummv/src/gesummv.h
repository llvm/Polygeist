#pragma once
#include <cmath>


extern "C" {
void kernel_gesummv(
		    double alpha,
		    double beta,
		    double A[ 250 + 0][250 + 0],
		    double B[ 250 + 0][250 + 0],
		    double tmp[ 250 + 0],
		    double x[ 250 + 0],
		    double y[ 250 + 0]);
}