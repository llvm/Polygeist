#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>
#include "hls_stream.h"

// Define tensor dimensions
#define B 4        // Batch size
#define N 100        // Sequence length
#define dk 128       // Key/Query dimension
#define dv 128       // Value dimension
#define D 128       // Dimension

#define Fac 32
#define DFac (D/Fac)
#define Fac_N 25
#define NFac_N (N/Fac_N)
#define FIFO_DEPTH 3


typedef ap_fixed<16, 5> fixed_t;
typedef ap_fixed<32, 8> wide_t;


void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], fixed_t Output[B][N][dv]);
