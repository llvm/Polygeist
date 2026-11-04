#pragma once
#include <cmath>


extern "C" {
void kernel_doitgen(  
		    double A[ 25 + 0][20 + 0][30 + 0],
		    double C4[ 30 + 0][30 + 0],
		    double sum[ 30 + 0]);
}