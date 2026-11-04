#pragma once
#include <cmath>


extern "C" {
void kernel_doitgen(  
		    double A[ 50 + 0][40 + 0][60 + 0],
		    double C4[ 60 + 0][60 + 0],
		    double sum[ 60 + 0]);
}