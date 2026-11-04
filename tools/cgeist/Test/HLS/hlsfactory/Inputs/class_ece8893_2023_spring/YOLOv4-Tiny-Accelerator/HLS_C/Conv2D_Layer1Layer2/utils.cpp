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
#include "conv.h"

//--------------------------------------------------------------------------
// Function to load an input tile block from from off-chip DRAM 
// to on-chip BRAM.
//
// TODO: This is an incomplete function that you need to modify  
//       to handle the border conditions appropriately while loading.
//--------------------------------------------------------------------------
void load_input_tile_block_from_DRAM_L1 (
    fm_t in_fm_buf[IN_BUF_DEPTH1][IN_BUF_HEIGHT1][IN_BUF_WIDTH1], 
    fm_t in_fm[1024][416][416], 
    int  ti, 
    int  tj,
    int  d 
)
{


    const int height_offset = ti * TILE_HEIGHT1;  
    const int width_offset  = tj * TILE_WIDTH1;
    const int depth_offset = d*IN_BUF_DEPTH1;

    const int P = PADDING1;

    INPUT_BUFFER_DEPTH:
    for(int c = 0; c < IN_BUF_DEPTH1; c++)
    {
        INPUT_BUFFER_HEIGHT:
        for(int i = 0; i < IN_BUF_HEIGHT1; i++)
        {
            INPUT_BUFFER_WIDTH:
            for(int j = 0; j < IN_BUF_WIDTH1; j++)
            {
                // TODO: Handle border features here
                //
                // Hint: Either load 0 or input feature into 
                //       the buffer based on border conditions
		
		      int x = height_offset - P + i;
	       	int y = width_offset - P + j;	
		if (x >= 0 && x < IN_FM_HEIGHT1 && y >= 0 && y < IN_FM_WIDTH1)
	                in_fm_buf[c][i][j] = in_fm[c+depth_offset][x][y]; // Just a placeholder
		else
			in_fm_buf[c][i][j] = (fm_t) 0;
            }
        }
    }
}

//--------------------------------------------------------------------------
// Function to load layer parameters (weights and bias) for convolution.
//
// This function can be used as-is in your design. However, you are free to
// add pragmas, restructure code, etc. depending on your way of optimization.
//--------------------------------------------------------------------------
void load_layer_params_from_DRAM_L1 (
    wt_t weight_buf[OUT_BUF_DEPTH1][IN_BUF_DEPTH1][KERNEL_HEIGHT1][KERNEL_WIDTH1],
    wt_t bias_buf[OUT_BUF_DEPTH1],
    wt_t weights[1024][1024][3][3],
    wt_t bias[1024],
    int  kernel_group,
    int d
)
{
    const int kernel_offset  = kernel_group * OUT_BUF_DEPTH1;
    const int depth_offset = d*IN_BUF_DEPTH1;

    WEIGHT_KERNEL_NUM:
    for(int f = 0; f < OUT_BUF_DEPTH1; f++)
    {
        WEIGHT_KERNEL_DEPTH:
        for(int c = 0; c < IN_BUF_DEPTH1; c++)
        {
            WEIGHT_KERNEL_HEIGHT:
            for(int kh = 0; kh < 3; kh++)
	        {
                WEIGHT_KERNEL_WIDTH:
	            for(int kw = 0; kw < 3; kw++)
	            {
	                weight_buf[f][c][kh][kw] = weights[kernel_offset + f][c + depth_offset][kh][kw];
                }
            }
        }
    }
    
    BIAS:
    for(int f = 0; f < OUT_BUF_DEPTH1; f++)
    {
        bias_buf[f] = bias[kernel_offset + f];
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
void store_output_tile_to_DRAM_L1 (
    fm_t out_fm[1024][416][416], 
    fm_t out_fm_buf[OUT_BUF_DEPTH1][OUT_BUF_HEIGHT1][OUT_BUF_WIDTH1], 
    int  ti,
    int  tj,
    int  kernel_group
)
{
    const int depth_offset  = kernel_group * OUT_BUF_DEPTH1;
    const int height_offset = ti * OUT_BUF_HEIGHT1;
    const int width_offset  = tj * OUT_BUF_WIDTH1;

    OUTPUT_BUFFER_DEPTH:
    for(int f = 0; f < OUT_BUF_DEPTH1; f++)
    {
        OUTPUT_BUFFER_HEIGHT:
        for(int i = 0; i < OUT_BUF_HEIGHT1; i++)
        {
            OUTPUT_BUFFER_WIDTH:
            for(int j = 0; j < OUT_BUF_WIDTH1; j++)
            {
                // ReLU in-place
                if(out_fm_buf[f][i][j] < (fm_t) 0)
                {
                    out_fm[depth_offset + f][height_offset + i][width_offset + j] = (fm_t(0.1) * out_fm_buf[f][i][j]);
                }
                else
                {
                    out_fm[depth_offset + f][height_offset + i][width_offset + j] = out_fm_buf[f][i][j];
                }
            }
        }
    }
}


// Convolution Layer 1 with a kernel size of 3x3
// Stride length of 2
// 

void conv_3x3_L1 (
    fm_t Y_buf[OUT_BUF_DEPTH1][OUT_BUF_HEIGHT1][OUT_BUF_WIDTH1], 
    fm_t X_buf[IN_BUF_DEPTH1][IN_BUF_HEIGHT1][IN_BUF_WIDTH1],                		
    wt_t W_buf[OUT_BUF_DEPTH1][IN_BUF_DEPTH1][KERNEL_HEIGHT1][KERNEL_WIDTH1],
    wt_t B_buf[OUT_BUF_DEPTH1]
)
{
	OUT_BUF_DEPTH_CHECK:
		for(int c = 0; c < OUT_BUF_DEPTH1; c++) {
	OUT_BUF_HEIGHT_CHECK:
			for (int i = 0; i < OUT_BUF_HEIGHT1; i++) {
			OUT_BUF_WIDTH_CHECK:
				for (int j = 0; j < OUT_BUF_WIDTH1; j++) {
					Y_buf[c][i][j] = 0;
               
				IN_BUF_DEPTH_CHECK:
					for (int l = 0; l < IN_BUF_DEPTH1; l++) {
					KERNEL_HEIGHT_CHECK:
						for (int h = 0; h < KERNEL_HEIGHT1; h++) {
							int x = i*STRIDE1 + h;
						KERNEL_WIDTH_CHECK:
							for (int w = 0; w < KERNEL_WIDTH1; w++) {
								int y = j*STRIDE1 + w;
								Y_buf[c][i][j] +=  W_buf[c][l][h][w]*((fm_t)X_buf[l][x][y]);
							}
						
						}
					}
				}
			}
		}
}


void save_partial_output_L1 (
	fm_t part_out_buf[OUT_BUF_DEPTH1][OUT_BUF_HEIGHT1][OUT_BUF_WIDTH1],
	fm_t out_buf[OUT_BUF_DEPTH1][OUT_BUF_HEIGHT1][OUT_BUF_WIDTH1],
	wt_t bias_buf[OUT_BUF_DEPTH1],
	int d
)
{
	for (int f = 0; f < OUT_BUF_DEPTH1; f++)
	{
		for (int h = 0; h < OUT_BUF_HEIGHT1; h++)
		{
			for (int w = 0; w < OUT_BUF_WIDTH1; w++)
			{
				if (d == 0)
					part_out_buf[f][h][w] = out_buf[f][h][w] + bias_buf[f];
				else
					part_out_buf[f][h][w] += out_buf[f][h][w];
			}
		}
	}


}


void tiled_conv_L1 (
    fm_t input_feature_map[1024][416][416],
    wt_t layer_weights[1024][1024][3][3],
    wt_t layer_bias[1024],
    fm_t output_feature_map[1024][416][416]
)
{
    //--------------------------------------------------------------------------
    // On-chip buffers
    // You should NOT modify the buffer dimensions!
    //--------------------------------------------------------------------------
    fm_t conv_in_buf[IN_BUF_DEPTH1][IN_BUF_HEIGHT1][IN_BUF_WIDTH1];
    wt_t conv_wt_buf[OUT_BUF_DEPTH1][IN_BUF_DEPTH1][KERNEL_HEIGHT1][KERNEL_WIDTH1];
    wt_t conv_bias_buf[OUT_BUF_DEPTH1];
    fm_t conv_out_buf[OUT_BUF_DEPTH1][OUT_BUF_HEIGHT1][OUT_BUF_WIDTH1] = {0};
   
    fm_t partial_out_buf[OUT_BUF_DEPTH1][OUT_BUF_HEIGHT1][OUT_BUF_WIDTH1] = {0};
    //--------------------------------------------------------------------------
    // Process each tile iteratively
    //--------------------------------------------------------------------------
    TILE_ROW:
    for(int ti = 0; ti < N_TILE_ROWS1; ti++)
    {
        TILE_COL:
        for(int tj = 0; tj < N_TILE_COLS1; tj++)
        {
            std::cout << "Processing Tile " << ti*N_TILE_COLS1 + tj + 1;
            std::cout << "/" << N_TILE_ROWS1 * N_TILE_COLS1 << std::endl;    

            //--------------------------------------------------------------------------
            // TODO: Your code for Task B and Task C goes here 
            //
            // Implement the required code to run convolution on an entire tile. 
            // Refer to utils.cpp for the required functions
            //
            // Hint: You need to split the filter kernels into sub-groups 
            //       for processing.
            //--------------------------------------------------------------------------
           
	    int kernel_group = OUT_FM_DEPTH1/OUT_BUF_DEPTH1;
	KERNEL_GROUP:
	    for (int k = 0; k < kernel_group; k++)
	    {
		    for (int d = 0; d < IN_FM_DEPTH1/IN_BUF_DEPTH1; d++)
		    {
	    		load_input_tile_block_from_DRAM_L1(conv_in_buf,input_feature_map,ti,tj,d);
		        load_layer_params_from_DRAM_L1(conv_wt_buf,conv_bias_buf,layer_weights, layer_bias,k,d);
			conv_3x3_L1(conv_out_buf,conv_in_buf,conv_wt_buf,conv_bias_buf);
			save_partial_output_L1(partial_out_buf,conv_out_buf,conv_bias_buf,d);
		    }
		store_output_tile_to_DRAM_L1(output_feature_map,partial_out_buf,ti,tj,k);

	    }

        }
    }
}


void load_input_tile_block_from_DRAM_L2 (
    fm_t in_fm_buf[IN_BUF_DEPTH2][IN_BUF_HEIGHT2][IN_BUF_WIDTH2], 
    fm_t in_fm[1024][416][416], 
    int  ti, 
    int  tj,
    int  d 
)
{
    const int height_offset = ti * TILE_HEIGHT2;  
    const int width_offset  = tj * TILE_WIDTH2;
    const int depth_offset = d*IN_BUF_DEPTH2;

    const int P = PADDING2;

    INPUT_BUFFER_DEPTH:
    for(int c = 0; c < IN_BUF_DEPTH2; c++)
    {
        INPUT_BUFFER_HEIGHT:
        for(int i = 0; i < IN_BUF_HEIGHT2; i++)
        {
            INPUT_BUFFER_WIDTH:
            for(int j = 0; j < IN_BUF_WIDTH2; j++)
            {
                // TODO: Handle border features here
                //
                // Hint: Either load 0 or input feature into 
                //       the buffer based on border conditions
		
		      int x = height_offset - P + i;
	       	int y = width_offset - P + j;	
		if (x >= 0 && x < IN_FM_HEIGHT2 && y >= 0 && y < IN_FM_WIDTH2)
	                in_fm_buf[c][i][j] = in_fm[c+depth_offset][x][y]; // Just a placeholder
		else
			in_fm_buf[c][i][j] = (fm_t) 0;
            }
        }
    }
}

//--------------------------------------------------------------------------
// Function to load layer parameters (weights and bias) for convolution.
//
// This function can be used as-is in your design. However, you are free to
// add pragmas, restructure code, etc. depending on your way of optimization.
//--------------------------------------------------------------------------
void load_layer_params_from_DRAM_L2 (
    wt_t weight_buf[OUT_BUF_DEPTH2][IN_BUF_DEPTH2][KERNEL_HEIGHT2][KERNEL_WIDTH2],
    wt_t bias_buf[OUT_BUF_DEPTH2],
    wt_t weights[1024][1024][3][3],
    wt_t bias[1024],
    int  kernel_group,
    int d
)
{
    const int kernel_offset  = kernel_group * OUT_BUF_DEPTH2;
    const int depth_offset = d*IN_BUF_DEPTH2;

    WEIGHT_KERNEL_NUM:
    for(int f = 0; f < OUT_BUF_DEPTH2; f++)
    {
        WEIGHT_KERNEL_DEPTH:
        for(int c = 0; c < IN_BUF_DEPTH2; c++)
        {
            WEIGHT_KERNEL_HEIGHT:
            for(int kh = 0; kh < 3; kh++)
	        {
                WEIGHT_KERNEL_WIDTH:
	            for(int kw = 0; kw < 3; kw++)
	            {
	                weight_buf[f][c][kh][kw] = weights[kernel_offset + f][c + depth_offset][kh][kw];
                }
            }
        }
    }
    
    BIAS:
    for(int f = 0; f < OUT_BUF_DEPTH2; f++)
    {
        bias_buf[f] = bias[kernel_offset + f];
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
void store_output_tile_to_DRAM_L2 (
    fm_t out_fm[1024][416][416], 
    fm_t out_fm_buf[OUT_BUF_DEPTH2][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2], 
    int  ti,
    int  tj,
    int  kernel_group
)
{
    const int depth_offset  = kernel_group * OUT_BUF_DEPTH2;
    const int height_offset = ti * OUT_BUF_HEIGHT2;
    const int width_offset  = tj * OUT_BUF_WIDTH2;

    OUTPUT_BUFFER_DEPTH:
    for(int f = 0; f < OUT_BUF_DEPTH2; f++)
    {
        OUTPUT_BUFFER_HEIGHT:
        for(int i = 0; i < OUT_BUF_HEIGHT2; i++)
        {
            OUTPUT_BUFFER_WIDTH:
            for(int j = 0; j < OUT_BUF_WIDTH2; j++)
            {
                // ReLU in-place
                if(out_fm_buf[f][i][j] < (fm_t) 0)
                {
                    out_fm[depth_offset + f][height_offset + i][width_offset + j] = (fm_t(0.1) * out_fm_buf[f][i][j]);
                }
                else
                {
                    out_fm[depth_offset + f][height_offset + i][width_offset + j] = out_fm_buf[f][i][j];
                }
            }
        }
    }
}


// Convolution Layer 1 with a kernel size of 3x3
// Stride length of 2
// 

void conv_3x3_L2 (
    fm_t Y_buf[OUT_BUF_DEPTH2][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2], 
    fm_t X_buf[IN_BUF_DEPTH2][IN_BUF_HEIGHT2][IN_BUF_WIDTH2],                		
    wt_t W_buf[OUT_BUF_DEPTH2][IN_BUF_DEPTH2][KERNEL_HEIGHT2][KERNEL_WIDTH2],
    wt_t B_buf[OUT_BUF_DEPTH2]
)
{
		OUT_BUF_DEPTH_CHECK:
 	      		for(int c = 0; c < OUT_BUF_DEPTH2; c++) {
        	OUT_BUF_HEIGHT_CHECK:
        			for (int i = 0; i < OUT_BUF_HEIGHT2; i++) {
        			OUT_BUF_WIDTH_CHECK:
        				for (int j = 0; j < OUT_BUF_WIDTH2; j++) {
        					Y_buf[c][i][j] = 0;
        				IN_BUF_DEPTH_CHECK:
        					for (int l = 0; l < IN_BUF_DEPTH2; l++) {
        					KERNEL_HEIGHT_CHECK:
        						for (int h = 0; h < KERNEL_HEIGHT2; h++) {
        							int x = i*STRIDE2 + h;
        						KERNEL_WIDTH_CHECK:
        							for (int w = 0; w < KERNEL_WIDTH2; w++) {
        								int y = j*STRIDE2 + w;
        								Y_buf[c][i][j] +=  W_buf[c][l][h][w]*((fm_t)X_buf[l][x][y]);
        							}
        						
        						}
        					}
        				}
        			}
        		}
        }


void save_partial_output_L2 (
	fm_t part_out_buf[OUT_BUF_DEPTH2][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2],
	fm_t out_buf[OUT_BUF_DEPTH2][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2],
	wt_t bias_buf[OUT_BUF_DEPTH2],
	int d
)
{
	for (int f = 0; f < OUT_BUF_DEPTH2; f++)
	{
		for (int h = 0; h < OUT_BUF_HEIGHT2; h++)
		{
			for (int w = 0; w < OUT_BUF_WIDTH2; w++)
			{
				if (d == 0)
					part_out_buf[f][h][w] = out_buf[f][h][w] + bias_buf[f];
				else
					part_out_buf[f][h][w] += out_buf[f][h][w];
			}
		}
	}


}


void tiled_conv_L2 (
    fm_t input_feature_map[1024][416][416],
    wt_t layer_weights[1024][1024][3][3],
    wt_t layer_bias[1024],
    fm_t output_feature_map[1024][416][416]
)
{
    //--------------------------------------------------------------------------
    // On-chip buffers
    // You should NOT modify the buffer dimensions!
    //--------------------------------------------------------------------------
    fm_t conv_in_ping_buf[IN_BUF_DEPTH2][IN_BUF_HEIGHT2][IN_BUF_WIDTH2];
    fm_t conv_in_pong_buf[IN_BUF_DEPTH2][IN_BUF_HEIGHT2][IN_BUF_WIDTH2];
    wt_t conv_wt_ping_buf[OUT_BUF_DEPTH2][IN_BUF_DEPTH2][KERNEL_HEIGHT2][KERNEL_WIDTH2];
    wt_t conv_wt_pong_buf[OUT_BUF_DEPTH2][IN_BUF_DEPTH2][KERNEL_HEIGHT2][KERNEL_WIDTH2];
    wt_t conv_bias_ping_buf[OUT_BUF_DEPTH2];
    wt_t conv_bias_pong_buf[OUT_BUF_DEPTH2];
    fm_t conv_out_buf[OUT_BUF_DEPTH2][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2] = {0};
   
    fm_t partial_out_buf[OUT_BUF_DEPTH2][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2] = {0};
    //--------------------------------------------------------------------------
    // Process each tile iteratively
    //--------------------------------------------------------------------------
    TILE_ROW:
    for(int ti = 0; ti < N_TILE_ROWS2; ti++)
    {
        TILE_COL:
        for(int tj = 0; tj < N_TILE_COLS2; tj++)
        {
            std::cout << "Processing Tile " << ti*N_TILE_COLS2 + tj + 1;
            std::cout << "/" << N_TILE_ROWS2 * N_TILE_COLS2 << std::endl;    

            //--------------------------------------------------------------------------
            // TODO: Your code for Task B and Task C goes here 
            //
            // Implement the required code to run convolution on an entire tile. 
            // Refer to utils.cpp for the required functions
            //
            // Hint: You need to split the filter kernels into sub-groups 
            //       for processing.
            //--------------------------------------------------------------------------
           
	    int kernel_group = OUT_FM_DEPTH2/OUT_BUF_DEPTH2;
	KERNEL_GROUP:
	    for (int k = 0; k < kernel_group; k++)
	    {

	    		load_input_tile_block_from_DRAM_L2(conv_in_ping_buf,input_feature_map,ti,tj,0);
		        load_layer_params_from_DRAM_L2(conv_wt_ping_buf,conv_bias_ping_buf,layer_weights, layer_bias,k,0);

		    for (int d = 0; d < (IN_FM_DEPTH2/IN_BUF_DEPTH2) - 1; d++)
		    {
			if (d % 2 == 0)
			{

			conv_3x3_L2(conv_out_buf,conv_in_ping_buf,conv_wt_ping_buf,conv_bias_ping_buf);
			save_partial_output_L2(partial_out_buf,conv_out_buf,conv_bias_ping_buf,d);

	    		load_input_tile_block_from_DRAM_L2(conv_in_pong_buf,input_feature_map,ti,tj,d+1);
		        load_layer_params_from_DRAM_L2(conv_wt_pong_buf,conv_bias_pong_buf,layer_weights, layer_bias,k,d+1);
			}
			else if (d % 2 == 1)
			{
			conv_3x3_L2(conv_out_buf,conv_in_pong_buf,conv_wt_pong_buf,conv_bias_pong_buf);
			save_partial_output_L2(partial_out_buf,conv_out_buf,conv_bias_pong_buf,d);
	    		load_input_tile_block_from_DRAM_L2(conv_in_ping_buf,input_feature_map,ti,tj,d+1);
		        load_layer_params_from_DRAM_L2(conv_wt_ping_buf,conv_bias_ping_buf,layer_weights, layer_bias,k,d+1);
			}
		    }
		    if (((IN_FM_DEPTH2/IN_BUF_DEPTH2)-1)%2 == 0)
		     {

			conv_3x3_L2(conv_out_buf,conv_in_ping_buf,conv_wt_ping_buf,conv_bias_ping_buf);
			save_partial_output_L2(partial_out_buf,conv_out_buf,conv_bias_ping_buf,(IN_FM_DEPTH2/IN_BUF_DEPTH2)-1);
		     }
		    else if (((IN_FM_DEPTH2/IN_BUF_DEPTH2)-1)%2 == 1)
		    {

			conv_3x3_L2(conv_out_buf,conv_in_pong_buf,conv_wt_pong_buf,conv_bias_pong_buf);
			save_partial_output_L2(partial_out_buf,conv_out_buf,conv_bias_pong_buf,(IN_FM_DEPTH2/IN_BUF_DEPTH2)-1);

		    }

		store_output_tile_to_DRAM_L2(output_feature_map,partial_out_buf,ti,tj,k);

	    }

        }
    }
}



void copy_output_to_input_L2 (
   fm_t input_feature_map[1024][416][416],
   fm_t output_feature_map[1024][416][416]
  )
{

	for (int c = 0; c < IN_FM_DEPTH2; c++)
	{
		for (int i = 0; i < IN_FM_HEIGHT2; i++)
		{
			for (int j = 0; j < IN_FM_WIDTH2; j++)
			{
				input_feature_map[c][i][j] = output_feature_map[c][i][j];
			}
		}
	}

}








