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

void kernel_mvt(double x1[120], double x2[120], double y_1[120], double y_2[120], double A[120][120]);

#endif
