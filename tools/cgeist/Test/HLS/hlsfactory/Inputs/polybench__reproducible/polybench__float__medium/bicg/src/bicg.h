#pragma once
#include <cmath>


extern "C" {
void kernel_bicg( 
		 double A[ 410 + 0][390 + 0],
		 double s[ 390 + 0],
		 double q[ 410 + 0],
		 double p[ 390 + 0],
		 double r[ 410 + 0]);
}