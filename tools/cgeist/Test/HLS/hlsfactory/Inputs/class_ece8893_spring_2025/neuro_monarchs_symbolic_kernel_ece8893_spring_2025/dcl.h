#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdlib.h>
#include <cstdint>
#include <ap_int.h>
#include <string.h>
#include <hls_stream.h>

typedef ap_uint<512> mem_t;
#define MS 16   // Memory size - Number of floats per mem_t

#define BS 16   // Batch size
#define K 4     // Rows of A, B, and C
#define L 256   // Columns of A, B, and C

#define PE 8    // Number of processing elements (power of 2)
#define UF 8    // Unroll factor (Mults, Accums, etc.) (power of 2)

#define RPPE (BS*K)/PE  // Rows per processing element

void symbolic_circular_conv_HLS(mem_t A_mem[BS][K][L/MS], mem_t B_mem[BS][K][L/MS], mem_t C_mem[BS][K][L/MS]);


