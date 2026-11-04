#pragma once
#include <cmath>


extern "C" {
void kernel_2mm(   
		double alpha,
		double beta,
		double tmp[ 40 + 0][50 + 0],
		double A[ 40 + 0][70 + 0],
		double B[ 70 + 0][50 + 0],
		double C[ 50 + 0][80 + 0],
		double D[ 40 + 0][80 + 0]);
}