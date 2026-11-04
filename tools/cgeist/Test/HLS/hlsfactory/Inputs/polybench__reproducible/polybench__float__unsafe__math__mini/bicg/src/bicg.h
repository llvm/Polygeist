#pragma once
#include <cmath>


extern "C" {
void kernel_bicg( 
		 double A[ 42 + 0][38 + 0],
		 double s[ 38 + 0],
		 double q[ 42 + 0],
		 double p[ 38 + 0],
		 double r[ 42 + 0]);
}