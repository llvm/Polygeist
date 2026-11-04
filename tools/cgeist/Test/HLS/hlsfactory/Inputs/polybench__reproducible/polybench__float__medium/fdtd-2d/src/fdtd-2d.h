#pragma once
#include <cmath>


extern "C" {
void kernel_fdtd_2d(
		    
		    
		    double ex[ 200 + 0][240 + 0],
		    double ey[ 200 + 0][240 + 0],
		    double hz[ 200 + 0][240 + 0],
		    double _fict_[ 100 + 0]);
}