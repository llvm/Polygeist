#pragma once
#include <cmath>


extern "C" {
void kernel_fdtd_2d(
		    
		    
		    double ex[ 60 + 0][80 + 0],
		    double ey[ 60 + 0][80 + 0],
		    double hz[ 60 + 0][80 + 0],
		    double _fict_[ 40 + 0]);
}