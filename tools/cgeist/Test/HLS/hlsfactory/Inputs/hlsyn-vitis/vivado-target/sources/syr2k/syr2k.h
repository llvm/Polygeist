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

void kernel_syr2k(double alpha,double beta,double C[80][80],double A[80][60],double B[80][60]);

#endif
