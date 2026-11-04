#pragma once
#include <cmath>


extern "C" {
void kernel_trisolv(
		    double L[ 40 + 0][40 + 0],
		    double x[ 40 + 0],
		    double b[ 40 + 0]);
}