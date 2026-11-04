#pragma once
#include <cmath>


extern "C" {
void kernel_bicg( 
		 double A[ 124 + 0][116 + 0],
		 double s[ 116 + 0],
		 double q[ 124 + 0],
		 double p[ 116 + 0],
		 double r[ 124 + 0]);
}