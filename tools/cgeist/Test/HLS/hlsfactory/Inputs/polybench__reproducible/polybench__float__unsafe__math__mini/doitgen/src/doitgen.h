#pragma once
#include <cmath>


extern "C" {
void kernel_doitgen(  
		    double A[ 10 + 0][8 + 0][12 + 0],
		    double C4[ 12 + 0][12 + 0],
		    double sum[ 12 + 0]);
}