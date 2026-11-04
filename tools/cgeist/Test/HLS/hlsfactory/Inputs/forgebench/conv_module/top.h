#ifndef COMMON_CONV
#define COMMON_CONV

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>
#include <hls_math.h>

// The configuration of global BRAM
#define MAX_C 128   // Maximum channels
#define MAX_H 28   // Maximum height
#define MAX_W 28   // Maximum width

// The configuration of the kernel of the feature map
#define MAX_LOCAL_SIZE 15 //stride*(BLOCK_OUT_H - 1) + KSIZE 2 * 6 + 3
#define BLOCK_IN_CH 64
#define BLOCK_OUT_CH 64
#define BLOCK_OUT_H 7  // maximum tile height (output)
#define BLOCK_OUT_W 7  // maximum tile width (output)
#define KSIZE 3


typedef ap_fixed<16, 5> data_t;

// Inline functions or function definitions in a header should be marked inline
int out_dim(int size, int pad, int stride, int kernel) {
    return (size + 2 * pad - kernel) / stride + 1;
}

int ceil_div(int x, int y) {
    return (x + y - 1) / y;
}

int min_value (int x, int y) {
    return (x < y) ? x : y;
}

#endif // COMMON_H

