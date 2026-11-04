
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>
#include <hls_stream.h>

typedef ap_fixed<16, 5> fixed_t;

// Data type for large accumulations
typedef ap_fixed<32, 8> accum_t;

// Data type for wide bus transactions
typedef ap_uint<512> mem_t;

// Define tensor dimensions
//#define B   32      // Batch size
#define B   4      // Batch size
#define N   100     // Sequence length
#define dk  128     // Key/Query dimension
#define dv  128     // Value dimension

void compute_attention_HLS(mem_t Q[B][N][dk/32], mem_t K[B][N][dk/32], mem_t V[B][N][dv/32], mem_t Output[B][N][dv/32]);
