///////////////////////////////////////////////////////////////////////////////
// Author:      <>
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    conv_7x7.cpp
// Description: Implement an optimized 7x7 convolution for a single tile block
//
// TODO: Use your unoptimized code from Part B and apply your favorite pragmas
//       to achieve the target latency (or lower)!
///////////////////////////////////////////////////////////////////////////////
#include "utils.h"

void conv_7x7(
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH],
    fm_t X_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH],
    wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    wt_t B_buf[OUT_BUF_DEPTH])
{
#pragma HLS inline off
#pragma HLS array_partition variable = X_buf dim = 2
#pragma HLS array_partition variable = Y_buf dim = 2
#pragma HLS array_partition variable = Y_buf dim = 1
#pragma HLS array_partition variable = W_buf dim = 1
#pragma HLS array_partition variable = B_buf dim = 1

INIT_OUTPUT_WIDTH:
    for (int j = 0; j < OUT_BUF_WIDTH; j++)
    {
#pragma HLS pipeline II = 1
    INIT_OUTPUT_HEIGHT:
        for (int i = 0; i < OUT_BUF_HEIGHT; i++)
        {
        INIT_OUTPUT_DEPTH:
            for (int f = 0; f < OUT_BUF_DEPTH; f++)
            {
                Y_buf[f][i][j] = B_buf[f];
            }
        }
    }

OUTPUT_WIDTH_CONV:
    for (int j = 0; j < OUT_BUF_WIDTH; j++) // Output Width
    {
    KERNEL_HEIGHT_CONV:
        for (int kh = 0; kh < KERNEL_HEIGHT; kh++) // Kernel Height
        {
        KERNEL_WIDTH_CONV:
            for (int kw = 0; kw < KERNEL_WIDTH; kw++) // Kernel Width
            {

            INPUT_DEPTH_CONV:
                for (int c = 0; c < IN_BUF_DEPTH; c++) // Input Depth
                {

#pragma HLS pipeline II = 1
                FILTER_DEPTH_CONV:
                    for (int f = 0; f < OUT_BUF_DEPTH; f++) // Filter Size (Output Depth)
                    {

                    OUTPUT_HEIGHT_CONV:
                        for (int i = 0; i < OUT_BUF_HEIGHT; i++) // Output Height

                            Y_buf[f][i][j] += X_buf[c][STRIDE * (i) + kh][STRIDE * (j) + kw] * W_buf[f][c][kh][kw];
                    }
                }
            }
        }
    }
}
