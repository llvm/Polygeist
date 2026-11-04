#pragma once
#include <cmath>


extern "C" {
void kernel_gramschmidt( 
			double A[ 20 + 0][30 + 0],
			double R[ 30 + 0][30 + 0],
			double Q[ 20 + 0][30 + 0]);
}