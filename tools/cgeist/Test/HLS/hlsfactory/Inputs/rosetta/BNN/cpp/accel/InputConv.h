#ifndef ACCEL_INPUT_CONV_H
#define ACCEL_INPUT_CONV_H

#include "Typedefs.h"
#include "Accel.h"

void run_input_conv_layer(
    const float* w_data,
    const float* k_data,
    const float* h_data,
    const float* data_i,
    Word* data_o,
    const unsigned M,
    const unsigned N
);

#endif
