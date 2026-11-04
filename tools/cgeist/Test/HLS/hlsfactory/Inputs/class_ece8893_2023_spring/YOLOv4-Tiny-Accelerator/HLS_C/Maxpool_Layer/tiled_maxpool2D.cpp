///////////////////////////////////////////////////////////////////////////////
// Author:      <>
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    tiled_conv.cpp
// Description: Implement a functionally-correct synthesizable tiling-based 
//              convolution for ResNet-50's first 7x7 layer with an 
//              HD input image.
//              
// Note: Do not use pragmas other than the existing ones in Part B.
///////////////////////////////////////////////////////////////////////////////
#include "utils.h"

void tiled_maxpool2D (
    fm_t input_feature_map[IN_FM_DEPTH][IN_FM_HEIGHT][IN_FM_WIDTH],
    fm_t output_feature_map[OUT_FM_DEPTH][OUT_FM_HEIGHT][OUT_FM_WIDTH]
)
{
    //--------------------------------------------------------------------------
    // Defines interface IO ports for HLS. 
    //--------------------------------------------------------------------------
    #pragma HLS INTERFACE m_axi depth=1  port=input_feature_map   bundle=fm
    #pragma HLS INTERFACE m_axi depth=1  port=output_feature_map  bundle=fm
    
    #pragma HLS INTERFACE s_axilite register	port=return
    
    //--------------------------------------------------------------------------
    // On-chip buffers
    // You should NOT modify the buffer dimensions!
    //--------------------------------------------------------------------------
    fm_t maxpool_in_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH];
    fm_t maxpool_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH] = {0};
    
    //--------------------------------------------------------------------------
    // Process each tile iteratively
    //--------------------------------------------------------------------------
    TILE_ROW:
    for(int ti = 0; ti < N_TILE_ROWS; ti++)
    {
        TILE_COL:
        for(int tj = 0; tj < N_TILE_COLS; tj++)
        {
            std::cout << "Processing Tile " << ti*N_TILE_COLS + tj + 1;
            std::cout << "/" << N_TILE_ROWS * N_TILE_COLS << std::endl;    

            //--------------------------------------------------------------------------
            // TODO: Your code for Task B and Task C goes here 
            //
            // Implement the required code to run convolution on an entire tile. 
            // Refer to utils.cpp for the required functions
            //
            // Hint: You need to split the filter kernels into sub-groups 
            //       for processing.
            //--------------------------------------------------------------------------
           
	    int kernel_group = OUT_FM_DEPTH/OUT_BUF_DEPTH;
	KERNEL_GROUP:
	    for (int k = 0; k < kernel_group; k++)
	    {
		for (int c = 0; c < IN_FM_DEPTH/IN_BUF_DEPTH; c++)
		{	
	    	load_input_tile_block_from_DRAM(maxpool_in_buf,input_feature_map,ti,tj,c);
		maxpool2D(maxpool_out_buf,maxpool_in_buf);
		store_output_tile_to_DRAM(output_feature_map,maxpool_out_buf,ti,tj,c);
		}
	    }

        }
    }
}
