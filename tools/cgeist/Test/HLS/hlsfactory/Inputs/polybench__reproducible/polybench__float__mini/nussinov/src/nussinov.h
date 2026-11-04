#pragma once
#include <cmath>


extern "C" {
void kernel_nussinov( char seq[ 60 + 0],
			   int table[ 60 + 0][60 + 0]);
}