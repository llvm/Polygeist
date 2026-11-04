#ifndef TOP_H
#define TOP_H


#include <string.h>
#include "ap_fixed.h"
#include <complex>
#include <hls_stream.h>
#include <iomanip>
#include <iostream>
#include <cmath>
#include "hls_vector.h"

void spmv(double val[1666],int cols[1666],int rowDelimiters[495],double vec[494],double out[494]);

#endif
