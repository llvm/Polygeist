#include "linear.h"

using std::array;

template<
    int DIM_IN,
    int DIM_OUT,
    int PARALLEL,
    bool RELU
>
void linear(
    FM_TYPE input[DIM_IN],
    WT_TYPE weight[DIM_OUT][DIM_IN],
    WT_TYPE bias[DIM_OUT],
    FM_TYPE output[DIM_OUT]
)
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=input complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight cyclic factor=PARALLEL dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=bias cyclic factor=PARALLEL dim=1
#pragma HLS ARRAY_PARTITION variable=output cyclic factor=PARALLEL dim=1

    for (int dim_out_base = 0; dim_out_base < DIM_OUT; dim_out_base += PARALLEL)
    {
#pragma HLS PIPELINE II=1
        for (int dim_out_offset = 0; dim_out_offset < PARALLEL; dim_out_offset++)
        {
#pragma HLS UNROLL
            int dim_out = dim_out_base + dim_out_offset;
            FM_TYPE out_el = 0;

            if (dim_out < DIM_OUT)
            {
                out_el = bias[dim_out];
                for (int dim_in = 0; dim_in < DIM_IN; dim_in++)
                {
#pragma HLS UNROLL
                    out_el += input[dim_in] * weight[dim_out][dim_in];
                }
            }

            if (RELU && hls::signbit(out_el)) out_el = 0;
            output[dim_out] = out_el;
        }
    }
}

template<
    int DIM_IN,
    int DIM_OUT,
    int PARALLEL,
    bool RELU
>
void linear_output_stationary(
    FM_TYPE input[DIM_IN],
    WT_TYPE weight[DIM_OUT][DIM_IN],
    WT_TYPE bias[DIM_OUT],
    hls::stream<array<FM_TYPE, PARALLEL>>& output
)
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=input complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight cyclic factor=PARALLEL dim=1
#pragma HLS ARRAY_PARTITION variable=weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=bias cyclic factor=PARALLEL dim=1

    for (int dim_out_base = 0; dim_out_base < DIM_OUT; dim_out_base += PARALLEL)
    {
#pragma HLS PIPELINE II=1
        array<FM_TYPE, PARALLEL> out_slice;
        for (int dim_out_offset = 0; dim_out_offset < PARALLEL; dim_out_offset++)
        {
#pragma HLS UNROLL
            int dim_out = dim_out_base + dim_out_offset;
            FM_TYPE out_el = 0;

            if (dim_out < DIM_OUT)
            {
                out_el = bias[dim_out];
                for (int dim_in = 0; dim_in < DIM_IN; dim_in++)
                {
#pragma HLS UNROLL
                    out_el += input[dim_in] * weight[dim_out][dim_in];
                }
            }

            if (RELU && hls::signbit(out_el)) out_el = 0;
            out_slice[dim_out_offset] = out_el;
        }
        output << out_slice;
    }
}

template<
    int DIM_IN,
    int DIM_OUT,
    int PARALLEL,
    bool RELU
>
void linear_input_stationary(
    hls::stream<array<FM_TYPE, PARALLEL>>& input,
    WT_TYPE weight[DIM_OUT][DIM_IN],
    WT_TYPE bias[DIM_OUT],
    FM_TYPE output[DIM_OUT]
)
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight cyclic factor=PARALLEL dim=2
#pragma HLS ARRAY_PARTITION variable=bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=output complete dim=1

    for (int dim_out = 0; dim_out < DIM_OUT; dim_out++)
    {
#pragma HLS UNROLL
        output[dim_out] = bias[dim_out];
    }

    for (int dim_in_base = 0; dim_in_base < DIM_IN; dim_in_base += PARALLEL)
    {
#pragma HLS PIPELINE II=1
        array<FM_TYPE, PARALLEL> in_slice;
        input >> in_slice;
        for (int dim_out = 0; dim_out < DIM_OUT; dim_out++)
        {
#pragma HLS UNROLL
            FM_TYPE addend = 0;
            for (int dim_in_offset = 0; dim_in_offset < PARALLEL; dim_in_offset++)
            {
#pragma HLS UNROLL
                int dim_in = dim_in_base + dim_in_offset;
                FM_TYPE in_el = in_slice[dim_in_offset];
                if (dim_in < DIM_IN)
                {
                    addend += in_el * weight[dim_out][dim_in];
                }
            }
            output[dim_out] += addend;
        }
    }

    for (int dim_out = 0; dim_out < DIM_OUT; dim_out++)
    {
#pragma HLS UNROLL
        if (RELU && hls::signbit(output[dim_out])) output[dim_out] = 0;
    }
}

// #region Template Instantiations

// From finalize.cc
template void linear_input_stationary<EMB_DIM, NUM_TASK, APPLY_PARALLEL, false>(
    hls::stream<array<FM_TYPE, APPLY_PARALLEL>>& input,
    WT_TYPE weight[NUM_TASK][EMB_DIM],
    WT_TYPE bias[NUM_TASK],
    FM_TYPE output[NUM_TASK]
);

// #endregion
