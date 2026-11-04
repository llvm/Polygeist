//---------------------------------------------------------------------------
// Perform synthesizable tiling-based convolution for a single tile.
//---------------------------------------------------------------------------
#include "conv.h"

void conv_3x3 (
    fm_t Y_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    fm_t X_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH],
    wt_t W_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][3][3]
)
{
//---------------------------------------------------------------------------
// Your code for Task B and Task C goes here. 
//
// Task B: Implement a trivial functionally-correct single-tile convolution.
//         This should have an overall latency in the order of 22-23 seconds.
//         If it's worse than trivial, first fix this!
//
// Task C: Optimize to achieve an overall latency of less than 950ms 
//---------------------------------------------------------------------------            

//Convolution computation
for(int f = 0; f < OUT_BUF_DEPTH; f++)                  // Filter Size (Output Depth)
    for(int i = 0; i < OUT_BUF_HEIGHT; i++)             // Output Height
        for(int j = 0; j < OUT_BUF_WIDTH; j++)          // Output Width
        {   
            Y_buf[f][i][j] = 0;                         //Initialize output feature
            for(int c = 0; c < IN_BUF_DEPTH; c++)       // Input Depth
                for(int kh = 0; kh < 3; kh++)           // Kernel Height
                    for(int kw = 0; kw < 3; kw++)       // Kernel Width		
                    {   // Multiple and Accumulate (MAC)
                        Y_buf[f][i][j] += X_buf[c][i+kh][j+kw] * W_buf[f][c][kh][kw];
                    }
        }

}
