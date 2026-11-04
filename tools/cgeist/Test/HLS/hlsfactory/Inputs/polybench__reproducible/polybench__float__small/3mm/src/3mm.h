#pragma once
#include <cmath>


extern "C" {
void kernel_3mm(    
		double E[ 40 + 0][50 + 0],
		double A[ 40 + 0][60 + 0],
		double B[ 60 + 0][50 + 0],
		double F[ 50 + 0][70 + 0],
		double C[ 50 + 0][80 + 0],
		double D[ 80 + 0][70 + 0],
		double G[ 40 + 0][70 + 0]);
}