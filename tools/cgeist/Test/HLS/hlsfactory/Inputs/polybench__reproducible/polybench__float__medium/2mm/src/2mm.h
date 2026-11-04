#pragma once
#include <cmath>


extern "C" {
void kernel_2mm(   
		double alpha,
		double beta,
		double tmp[ 180 + 0][190 + 0],
		double A[ 180 + 0][210 + 0],
		double B[ 210 + 0][190 + 0],
		double C[ 190 + 0][220 + 0],
		double D[ 180 + 0][220 + 0]);
}