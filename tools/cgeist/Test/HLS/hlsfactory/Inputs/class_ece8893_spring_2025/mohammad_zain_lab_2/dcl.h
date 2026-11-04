
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>
#include "hls_vector.h"
#include "hls_stream.h"

typedef ap_fixed<16, 5> fixed_t; //16 bits with 5 integer bits
typedef ap_uint<512> wide_bus_t; //512 bits -> 32 elements of 16 bits
// Define tensor dimensions
//Dont change these values
#define B   4      // Batch size 4
#define N  100     // Sequence length 10
#define dk 128     // Key/Query dimension 16
#define dv 128     // Value dimension 16

void compute_attention_HLS(wide_bus_t Q[B][N][dk/32], wide_bus_t K[B][N][dk/32], wide_bus_t V[B][N][dv/32], fixed_t Output[B][N][dv]);
