#pragma once
#include <cmath>


extern "C" {
void kernel_atax( 
		 double A[ 390 + 0][410 + 0],
		 double x[ 410 + 0],
		 double y[ 410 + 0],
		 double tmp[ 390 + 0]);
}