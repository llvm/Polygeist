#pragma once
#include <cmath>


extern "C" {
void kernel_gramschmidt( 
			double A[ 200 + 0][240 + 0],
			double R[ 240 + 0][240 + 0],
			double Q[ 200 + 0][240 + 0]);
}