#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>
#include <hls_math.h>

#ifndef TOP_H
#define TOP_H


typedef ap_fixed<16, 5> data_t;

#define C 64
#define H 28
#define W 28

void top(
    data_t input_sigmoid[C][H][W],
    data_t output_sigmoid[C][H][W], 
    data_t input_tanh[C][H][W],
    data_t output_tanh[C][H][W],
    data_t input_elu[C][H][W],
    data_t output_elu[C][H][W]
);
#endif // TOP_H