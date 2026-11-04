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

// Define tensor dimensions
#define B   4    // Batch size
#define N   100  // Sequence length
#define dk  128  // Key/Query dimension
#define dv  128  // Value dimension
#define fd  64   // Partition or Unroll factor for 128
#define fn  4    // Partition or Unroll factor for Softmax

