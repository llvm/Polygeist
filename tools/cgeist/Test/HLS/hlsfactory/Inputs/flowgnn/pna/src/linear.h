#ifndef __LINEAR_H__
#define __LINEAR_H__

#include "dcl.h"
#include "hls_stream.h"

template<
    int DIM_IN,
    int DIM_OUT,
    int PARALLEL,
    bool RELU = true
>
void linear(
    FM_TYPE input[DIM_IN],
    WT_TYPE weight[DIM_OUT][DIM_IN],
    WT_TYPE bias[DIM_OUT],
    FM_TYPE output[DIM_OUT]
);

template<
    int DIM_IN,
    int DIM_OUT,
    int PARALLEL,
    bool RELU = true
>
void linear_output_stationary(
    FM_TYPE input[DIM_IN],
    WT_TYPE weight[DIM_OUT][DIM_IN],
    WT_TYPE bias[DIM_OUT],
    hls::stream<std::array<FM_TYPE, PARALLEL>>& output
);

template<
    int DIM_IN,
    int DIM_OUT,
    int PARALLEL,
    bool RELU = true
>
void linear_input_stationary(
    hls::stream<std::array<FM_TYPE, PARALLEL>>& input,
    WT_TYPE weight[DIM_OUT][DIM_IN],
    WT_TYPE bias[DIM_OUT],
    FM_TYPE output[DIM_OUT]
);

#endif
