
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>
#include <hls_stream.h>
#include <hls_vector.h>

#define DATA_WIDTH 16
#define DATA_INT 5

typedef ap_fixed<DATA_WIDTH, DATA_INT> fixed_t;

// Define tensor dimensions
#define B   1      // Batch size
#define N   100     // Sequence length
#define dk  128     // Key/Query dimension
#define dv  128     // Value dimension

void compute_attention_HLS(hls::vector<fixed_t, dk> Q[B][N], hls::vector<fixed_t, dk> K[B][N], hls::vector<fixed_t, dv> V[B][N], hls::vector<fixed_t, dv> Output[B][N]);
