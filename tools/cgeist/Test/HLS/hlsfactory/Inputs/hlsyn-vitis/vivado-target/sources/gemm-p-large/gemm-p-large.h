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

void kernel_gemm(int ni,int nj,int nk,double alpha,double beta,double C[200][220],double A[200][240],double B[240][220]);

#endif
