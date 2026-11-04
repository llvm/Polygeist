//---------------------------------------------------------------------------
// Perform synthesizable tiling-based convolution for a single tile.
//---------------------------------------------------------------------------
#include "conv.h"

void conv_1x1 (
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2], 
    fm_t X_buf[IN_BUF_DEPTH2][IN_BUF_HEIGHT2][IN_BUF_WIDTH2],
    wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH2][1][1]
)
{
//#pragma HLS INLINE off
//---------------------------------------------------------------------------
// Your code for Task B and Task C goes here. 
//
// Task B: Implement a trivial functionally-correct single-tile convolution.
//         This should have an overall latency in the order of 22-23 seconds.
//         If it's worse than trivial, first fix this!
//
// Task C: Optimize to achieve an overall latency of less than 950ms 
//---------------------------------------------------------------------------            

#pragma HLS array_partition variable=X_buf dim=1
#pragma HLS array_partition variable=W_buf dim=2

//Convolution computation
for(int f = 0; f < OUT_BUF_DEPTH; f++)                  // Filter Size (Output Depth)
    for(int i = 0; i < OUT_BUF_HEIGHT2; i++)             // Output Height
        for(int j = 0; j < OUT_BUF_WIDTH2; j++)          // Output Width
        {   
#pragma HLS pipeline II=1
            Y_buf[f][i][j] = 0;                         //Initialize output feature
            fm_t Y_buf_local[IN_BUF_DEPTH2];
            for(int c = 0; c < IN_BUF_DEPTH2; c++)       // Input Depth
            {
#pragma HLS unroll
                Y_buf_local[c] = X_buf[c][i][j] * W_buf[f][c][0][0]; // Multiple and Accumulate (MAC)
            }
            for (int k = 0; k < IN_BUF_DEPTH2; k++)
            {
#pragma HLS unroll
                Y_buf[f][i][j] += Y_buf_local[k];
            }
        }
}
