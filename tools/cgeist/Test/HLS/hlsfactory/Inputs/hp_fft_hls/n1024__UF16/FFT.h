#ifndef FFT_H
#define FFT_H

#include "ap_fixed.h"
#include "hls_fft.h"
#include <complex>
#include <hls_stream.h>
#include <iomanip>   // std::setprecision, std::setw
#include <iostream>  // std::cout, std::fixed
#include <cmath>
#include "hls_vector.h"
#include "hls_streamofblocks.h"

using namespace std;
typedef float dtype_test;
typedef double dtype_gold;
#define batch_size 1 
#define FFT_NUM 1024
#define EXP2_FFT 10

#define UF 16


#ifndef PI
#define PI  3.14159265358979323846
#endif

void FFT_TOP(hls::stream<hls::vector<complex<float>, UF*2>> & in, 
    hls::stream<hls::vector<complex<float>, UF*2>> & out
);


#endif
