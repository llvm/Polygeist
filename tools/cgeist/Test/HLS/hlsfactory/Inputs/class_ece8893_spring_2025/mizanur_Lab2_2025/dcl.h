
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>

typedef ap_fixed<16, 5> fixed_t;

// Define tensor dimensions
#define B   4      // Batch size 4
#define N   100     // Sequence length 100
#define dk  128     // Key/Query dimension 128
#define dv  128     // Value dimension  128



void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], fixed_t Output[B][N][dv]);
