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

void kernel_gesummv(double alpha,double beta,double A[250][250],double B[250][250],double tmp[250],double x[250],double y[250]);

#endif
