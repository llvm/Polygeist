///////////////////////////////////////////////////////////////////////////////
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    conv_7x7.cpp
// Description: Reference implementation of 7x7 convolution for a
//              single tile block.
//
// TODO: Apply pragma(s) appropriately to accelerate computation!
///////////////////////////////////////////////////////////////////////////////
#include "utils.h"

void conv_7x7(
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH],
    fm_t X_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH],
    wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    wt_t B_buf[OUT_BUF_DEPTH])
{

#pragma HLS array_partition variable = X_buf dim = 2
#pragma HLS array_partition variable = Y_buf dim = 2
#pragma HLS array_partition variable = Y_buf dim = 1
#pragma HLS array_partition variable = W_buf dim = 1
#pragma HLS array_partition variable = B_buf dim = 1
BOW:
    for (int j = 0; j < TILE_WIDTH; j += STRIDE)
#pragma HLS pipeline II = 1
    BOH:
        for (int i = 0; i < TILE_HEIGHT; i += STRIDE)

        BOD:
            for (int f = 0; f < OUT_BUF_DEPTH; f++)

                Y_buf[f][i / STRIDE][j / STRIDE] = B_buf[f];

ID:
    for (int c = 0; c < IN_BUF_DEPTH; c++)
    {
    KH:
        for (int m = 0; m < KERNEL_HEIGHT; m++)
        {
        KW:
            for (int n = 0; n < KERNEL_WIDTH; n++)
            {
            OW:
                for (int j = 0; j < TILE_WIDTH; j += STRIDE)
                {
#pragma HLS pipeline II = 1
                OH:
                    for (int i = 0; i < TILE_HEIGHT; i += STRIDE)
                    {
                    OD:
                        for (int f = 0; f < OUT_BUF_DEPTH; f++)
                        {
                            Y_buf[f][i / STRIDE][j / STRIDE] += X_buf[c][i + m][j + n] * W_buf[f][c][m][n];
                        }
                    }
                }
            }
        }
    }
}
