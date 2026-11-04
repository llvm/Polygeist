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

void kernel_gemm(int ni,int nj,int nk,double alpha,double beta,double C[60][70],double A[60][80],double B[80][70]);

#endif
