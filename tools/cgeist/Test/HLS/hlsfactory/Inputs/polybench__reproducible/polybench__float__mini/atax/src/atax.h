#pragma once
#include <cmath>


extern "C" {
void kernel_atax( 
		 double A[ 38 + 0][42 + 0],
		 double x[ 42 + 0],
		 double y[ 42 + 0],
		 double tmp[ 38 + 0]);
}