//---------------------------------------------------------------------------
// Perform synthesizable tiling-based convolution for a single tile.
//---------------------------------------------------------------------------
#include "conv.h"

void conv_3x3_id16 (
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    fm_t X_buf[IN_BUF_DEPTH2][IN_BUF_HEIGHT][IN_BUF_WIDTH],
    wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH2][3][3]
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
#pragma HLS array_partition variable=X_buf dim=2 factor=3 cyclic
#pragma HLS array_partition variable=X_buf dim=3 factor=3 cyclic
#pragma HLS array_partition variable=W_buf dim=3
#pragma HLS array_partition variable=W_buf dim=4

//Convolution computation
for(int f = 0; f < OUT_BUF_DEPTH; f++)                  // Filter Size (Output Depth)
    for(int i = 0; i < OUT_BUF_HEIGHT; i++)             // Output Height
        for(int j = 0; j < OUT_BUF_WIDTH; j++)          // Output Width
        {   
#pragma HLS pipeline II=1
            Y_buf[f][i][j] = 0;                         //Initialize output feature
            fm_t Y_buf_local[IN_BUF_DEPTH2];
            for(int kh = 0; kh < 3; kh++)           // Kernel Height
            {
#pragma HLS unroll
                for(int kw = 0; kw < 3; kw++)       // Kernel Width	
                {
#pragma HLS unroll
                    for(int c = 0; c < IN_BUF_DEPTH2; c++)       // Input Depth
                    {
#pragma HLS unroll
                        Y_buf_local[c] = X_buf[c][i+kh][j+kw] * W_buf[f][c][kh][kw]; // Multiple and Accumulate (MAC)
                    }
                    for (int k = 0; k < IN_BUF_DEPTH2; k++)
                    {
#pragma HLS unroll
                        Y_buf[f][i][j] += Y_buf_local[k];
                    }
                }
            }
        }
        
}
