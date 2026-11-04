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

void kernel_gesummv(int n,double alpha,double beta,double A[90][90],double B[90][90],double tmp[90],double x[90],double y[90]);

#endif
