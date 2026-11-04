
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdlib.h>
#include <cstdint>

#include <ap_fixed.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>

typedef ap_fixed<16, 5> fixed_t;

// Define tensor dimensions
constexpr int B = 4;      // Batch size
constexpr int N = 100;     // Sequence length
constexpr int dk = 128;     // Key/Query dimension
constexpr int dv = 128;     // Value dimension

// Constants
constexpr int NUM_PACKED_VEC = 32; 
constexpr int NUM_VECS = dk / NUM_PACKED_VEC;
constexpr int NUM_PACKED_TILE = 10;
constexpr int NUM_TILES = N / NUM_PACKED_TILE;
constexpr float MIN = -1.0 * (1 << fixed_t::iwidth);

// Using
using packed_32_fixed_t = hls::vector<fixed_t, NUM_PACKED_VEC>; 
using tile_fixed_t = hls::vector<fixed_t, NUM_PACKED_TILE>;

void compute_attention_HLS(packed_32_fixed_t Q[B][N][NUM_VECS], packed_32_fixed_t K[B][N][NUM_VECS], tile_fixed_t V[B][dv][NUM_TILES], fixed_t Output[B][N][dv]);
