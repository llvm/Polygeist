
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <hls_stream.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>

typedef ap_fixed<16, 5> fixed_t;
typedef ap_fixed<16, 5> q_fixed_t;  // Not used due to higher error
typedef ap_uint<512> wide_t; // 512-bit wide data transfer

// Define tensor dimensions
#define B   4      // Batch size
#define N   100     // Sequence length
#define dk  128     // Key/Query dimension
#define dv  128     // Value dimension

void compute_attention_HLS(wide_t Q[B][N][dk / 32], wide_t K[B][N][dk / 32], wide_t V[B][N][dv / 32], wide_t Output[B][N][dv / 32]);