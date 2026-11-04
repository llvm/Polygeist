#pragma once
#include <cmath>


extern "C" {
void kernel_3mm(    
		double E[ 180 + 0][190 + 0],
		double A[ 180 + 0][200 + 0],
		double B[ 200 + 0][190 + 0],
		double F[ 190 + 0][210 + 0],
		double C[ 190 + 0][220 + 0],
		double D[ 220 + 0][210 + 0],
		double G[ 180 + 0][210 + 0]);
}