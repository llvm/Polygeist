#pragma once
#include <cmath>


extern "C" {
void kernel_2mm(   
		double alpha,
		double beta,
		double tmp[ 16 + 0][18 + 0],
		double A[ 16 + 0][22 + 0],
		double B[ 22 + 0][18 + 0],
		double C[ 18 + 0][24 + 0],
		double D[ 16 + 0][24 + 0]);
}