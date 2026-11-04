
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <stdlib.h>
#include <cstdint>

typedef ap_fixed<16, 5> fixed_t;

// Define tensor dimensions
#define B   4      // Batch size
#define N   100     // Sequence length
#define dk  128     // Key/Query dimension
#define dv  128     // Value dimension

// #define B   8      // Batch size
// #define N   10     // Sequence length
// #define dk  32     // Key/Query dimension
// #define dv  32     // Value dimension
#define TF  16
#define BurstSize 32   

typedef hls::vector<fixed_t, BurstSize> fixed_vector_t;

void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], fixed_t Output[B][N][dv]);
