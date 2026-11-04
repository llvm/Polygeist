#pragma once
#include <cmath>


extern "C" {
void kernel_gramschmidt( 
			double A[ 60 + 0][80 + 0],
			double R[ 80 + 0][80 + 0],
			double Q[ 60 + 0][80 + 0]);
}