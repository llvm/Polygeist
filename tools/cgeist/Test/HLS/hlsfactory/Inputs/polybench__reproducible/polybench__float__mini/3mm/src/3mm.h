#pragma once
#include <cmath>


extern "C" {
void kernel_3mm(    
		double E[ 16 + 0][18 + 0],
		double A[ 16 + 0][20 + 0],
		double B[ 20 + 0][18 + 0],
		double F[ 18 + 0][22 + 0],
		double C[ 18 + 0][24 + 0],
		double D[ 24 + 0][22 + 0],
		double G[ 16 + 0][22 + 0]);
}