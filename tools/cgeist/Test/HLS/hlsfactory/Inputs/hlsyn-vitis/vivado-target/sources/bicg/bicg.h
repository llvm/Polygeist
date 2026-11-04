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

void kernel_bicg(int m,int n,double A[124][116],double s[116],double q[124],double p[116],double r[124]);

#endif
