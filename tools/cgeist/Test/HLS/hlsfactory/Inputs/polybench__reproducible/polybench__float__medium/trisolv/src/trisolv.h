#pragma once
#include <cmath>


extern "C" {
void kernel_trisolv(
		    double L[ 400 + 0][400 + 0],
		    double x[ 400 + 0],
		    double b[ 400 + 0]);
}