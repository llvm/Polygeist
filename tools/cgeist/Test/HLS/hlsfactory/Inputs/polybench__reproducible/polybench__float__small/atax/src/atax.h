#pragma once
#include <cmath>


extern "C" {
void kernel_atax( 
		 double A[ 116 + 0][124 + 0],
		 double x[ 124 + 0],
		 double y[ 124 + 0],
		 double tmp[ 116 + 0]);
}