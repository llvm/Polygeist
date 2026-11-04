#pragma once
#include <cmath>


extern "C" {
void kernel_trisolv(
		    double L[ 120 + 0][120 + 0],
		    double x[ 120 + 0],
		    double b[ 120 + 0]);
}