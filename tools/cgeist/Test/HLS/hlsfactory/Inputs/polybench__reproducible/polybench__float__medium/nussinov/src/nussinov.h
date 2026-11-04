#pragma once
#include <cmath>


extern "C" {
void kernel_nussinov( char seq[ 500 + 0],
			   int table[ 500 + 0][500 + 0]);
}