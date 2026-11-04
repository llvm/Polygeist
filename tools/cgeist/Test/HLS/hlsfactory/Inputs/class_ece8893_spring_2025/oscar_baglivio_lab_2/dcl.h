
#include <ap_fixed.h>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <hls_math.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

typedef ap_fixed<16, 5> fixed_t;
typedef ap_fixed<32, 8> fixed_large_t;

// Define tensor dimensions
#define B 4    // Batch size
#define N 100  // Sequence length //3
#define dk 128 // Key/Query dimension //3
#define dv 128 // Value dimension //3

void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk],
                           fixed_t V[B][N][dv], fixed_t Output[B][N][dv]);
