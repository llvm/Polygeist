#ifndef ACCEL_DENSE_H
#define ACCEL_DENSE_H

#include "Debug.h"
#include "Typedefs.h"
#include "Accel.h"

void dense_layer_cpu(
    const Word* w,
    const float* k_data,
    const float* h_data,
    const Word* data_i,
    Word* data_o,
    const unsigned M,
    const unsigned N
);

int last_layer_cpu(
    const Word* w,
    const float* k_data,
    const float* h_data,
    const Word* in,
    const unsigned M,
    const unsigned N
);

#endif
