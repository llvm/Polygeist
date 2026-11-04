
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <hls_vector.h>
#include <hls_stream.h>
#include <stdlib.h>
#include <cstdint>


typedef ap_fixed<16, 5> fixed_t;

typedef hls::vector<fixed_t, 32> vec32_t;  // 32 fixed_t = 512-bit

// Define tensor dimensions
#define B   4      // Batch size
#define N   100     // Sequence length
#define dk  128     // Key/Query dimension
#define dv  128     // Value dimension

// // Change: Scale down for pre-try
// #define B   8      // Batch size
// #define N   2      // Sequence length
// #define dk  128     // Key/Query dimension
// #define dv  128     // Value dimension


void compute_attention_HLS(vec32_t Q[B][N][dk/32], vec32_t K[B][N][dk/32], vec32_t V[B][N][dv/32], vec32_t Output[B][N][dv/32]);

//void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], fixed_t Output[B][N][dv]);
//void compute_attention_HLS(vec32_t Q[B][N][dk/32], vec32_t K[B][N][dk/32], fixed_t V[B][N][dv], fixed_t Output[B][N][dv]);
