///////////////////////////////////////////////////////////////////////////////
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    utils.cpp
// Description: Utility functions to implement tiling-based convolution 
//
// Note:        Modify/complete the functions as required by your design. 
//
//              You are free to create any additional functions you may need
//              for your implementation.
///////////////////////////////////////////////////////////////////////////////
#include "utils.h"

//--------------------------------------------------------------------------
// Function to load an input tile block from from off-chip DRAM 
// to on-chip BRAM.
//
// TODO: This is an incomplete function that you need to modify  
//       to handle the border conditions appropriately while loading.
//--------------------------------------------------------------------------
void load_input_tile_block_from_DRAM (
    fm_t in_fm_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH], 
    fm_t in_fm[IN_FM_DEPTH][IN_FM_HEIGHT][IN_FM_WIDTH], 
    int  ti, 
    int  tj,
    int kernel_group 
)
{
    const int height_offset = ti * TILE_HEIGHT;  
    const int width_offset  = tj * TILE_WIDTH;

    const int depth_offset = kernel_group*IN_BUF_DEPTH;

    const int P = 0;

    INPUT_BUFFER_DEPTH:
    for(int c = 0; c < IN_BUF_DEPTH; c++)
    {
        INPUT_BUFFER_HEIGHT:
        for(int i = 0; i < IN_BUF_HEIGHT; i++)
        {
            INPUT_BUFFER_WIDTH:
            for(int j = 0; j < IN_BUF_WIDTH; j++)
            {
                // TODO: Handle border features here
                //
                // Hint: Either load 0 or input feature into 
                //       the buffer based on border conditions
		
		      int x = height_offset - P + i;
	       	int y = width_offset - P + j;	
	                in_fm_buf[c][i][j] = in_fm[c + depth_offset][x][y]; // Just a placeholder
            }
        }
    }
}

//--------------------------------------------------------------------------
// Function to store complete output tile block from BRAM to DRAM.
//
// In-place ReLU has been incorporated for convenience.
//
// This function can be used as-is in your design. However, you are free to
// add pragmas, restructure code, etc. depending on your way of optimization.
//--------------------------------------------------------------------------
void store_output_tile_to_DRAM (
    fm_t out_fm[OUT_FM_DEPTH][OUT_FM_HEIGHT][OUT_FM_WIDTH], 
    fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    int  ti,
    int  tj,
    int  kernel_group
)
{
    const int depth_offset  = kernel_group * OUT_BUF_DEPTH;
    const int height_offset = ti * OUT_BUF_HEIGHT;
    const int width_offset  = tj * OUT_BUF_WIDTH;

    OUTPUT_BUFFER_DEPTH:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        OUTPUT_BUFFER_HEIGHT:
        for(int i = 0; i < OUT_BUF_HEIGHT; i++)
        {
            OUTPUT_BUFFER_WIDTH:
            for(int j = 0; j < OUT_BUF_WIDTH; j++)
            {
                    out_fm[depth_offset + f][height_offset + i][width_offset + j] = out_fm_buf[f][i][j];
            }
        }
    }
}
