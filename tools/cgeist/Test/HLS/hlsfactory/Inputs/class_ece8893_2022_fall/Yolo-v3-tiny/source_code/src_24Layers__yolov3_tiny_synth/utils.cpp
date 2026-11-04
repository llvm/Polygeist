#include "conv.h"
#include <cmath>

void max_pool_2D(
    fm_t in_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH],
    fm_t out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT/2][OUT_BUF_WIDTH/2]
)
{
#pragma HLS INLINE off
    fm_t max;
    MAXPOOL_DEPTH:
    for (int d = 0; d < OUT_BUF_DEPTH; d++)
    {
        MAXPOOL_HEIGHT:
        for(int h = 0; h < OUT_BUF_HEIGHT/2; h++)
        {
            MAXPOOL_WIDTH:
            for(int w = 0; w < OUT_BUF_WIDTH/2; w++)
            {
                max = in_buf[d][2*h][2*w];
                MAXPOOL_WIN_DIM1:
                for (int i = 0; i < 2; i++)
                {
                    MAXPOOL_WIN_DIM2:
                    for(int j = 0; j < 2; j++)
                    {
                        if (in_buf[d][(2*h)+i][(2*w)+j] > max)
                            max = in_buf[d][(2*h)+i][(2*w)+j];
                    }
                }
                out_buf[d][h][w] = max;
            }
        }
    }
}

void max_pool_2D_stride1(
    fm_t in_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2+1][OUT_BUF_WIDTH2+1],
    fm_t out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2]
)
{
//#pragma HLS INLINE off
    fm_t max;
    MAXPOOL_DEPTH_STRIDE1:
    for (int d = 0; d < OUT_BUF_DEPTH; d++)
    {
        MAXPOOL_HEIGHT_STRIDE1:
        for (int h = 0; h < OUT_BUF_HEIGHT2; h++)
        {
            MAXPOOL_WIDTH_STRIDE1:
            for (int w = 0; w < OUT_BUF_WIDTH2; w++)
            {
                max = in_buf[d][h][w];
                MAXPOOL_WIN_DIM1_STRIDE1:
                for (int i = 0; i < 2; i++)
                {
                    MAXPOOL_WIN_DIM2_STRIDE1:
                    for (int j = 0; j < 2; j++)
                    {
                        if (in_buf[d][h+i][w+j] > max)
                            max = in_buf[d][h+i][w+j];
                    }
                }
                out_buf[d][h][w] = max;
            }
        }
    }
}

void load_maxpool_input_tile_block_from_DRAM(
    fm_t maxpool_in_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2+1][OUT_BUF_WIDTH2+1],
    fm_t input_mp_feature_map[1024][416][416],
    int d
)
{
    const int depth_offset  =  d * OUT_BUF_DEPTH;
        
    MAXPOOL_INPUT_BUFFER_DEPTH:
    for(int c = 0; c < OUT_BUF_DEPTH; c++)
    {
        MAXPOOL_INPUT_BUFFER_HEIGHT:
        for(int i = 0; i < OUT_BUF_HEIGHT2+1; i++)
        {
            MAXPOOL_INPUT_BUFFER_WIDTH:
            for(int j = 0; j < OUT_BUF_WIDTH2+1; j++)
            {
                // Handling border features here
                if((i == OUT_BUF_HEIGHT2) || (j == OUT_BUF_WIDTH2))
                    maxpool_in_buf[c][i][j] = 0;
                else
		            maxpool_in_buf[c][i][j] = input_mp_feature_map[depth_offset + c][i][j];
            }
        }
    }
}

void load_upsample_input_tile_block_from_DRAM(
    fm_t upsample_in_buf[US_BUF_DEPTH][US_BUF_HEIGHT][US_BUF_WIDTH],
    fm_t input_us_feature_map[1024][416][416],
    int d
)
{
    const int depth_offset  =  d * US_BUF_DEPTH;
        
    UPSAMPLE_INPUT_BUFFER_DEPTH:
    for(int c = 0; c < US_BUF_DEPTH; c++)
    {
        UPSAMPLE_INPUT_BUFFER_HEIGHT:
        for(int i = 0; i < US_BUF_HEIGHT; i++)
        {
            UPSAMPLE_INPUT_BUFFER_WIDTH:
            for(int j = 0; j < US_BUF_WIDTH; j++)
            {
		        upsample_in_buf[c][i][j] = input_us_feature_map[depth_offset + c][i][j];
            }
        }
    }
}

void upsample_2D(
    fm_t in_buf[US_BUF_DEPTH][US_BUF_HEIGHT][US_BUF_WIDTH],
    fm_t out_buf[US_BUF_DEPTH][US_BUF_HEIGHT*2][US_BUF_WIDTH*2]
)
{
    UPSAMPLE_BUFFER_DEPTH:
    for (int d = 0; d < US_BUF_DEPTH; d++)
    {
        UPSAMPLE_BUFFER_HEIGHT:
        for(int h = 0; h < US_BUF_HEIGHT; h++)
        {
            UPSAMPLE_BUFFER_WIDTH:
            for (int w = 0; w < US_BUF_WIDTH; w++)
            {
                UPSAMPLE_BUFFER_WIN_DIM1:
                for (int i = 0; i < 2; i++)
                {
                    UPSAMPLE_BUFFER_WIN_DIM2:
                    for(int j = 0; j < 2; j++)
                    {
                        out_buf[d][(2*h) + i][(2*w) + j] = in_buf[d][h][w];
                    }
                }
            }
        }
    }
}

//--------------------------------------------------------------------------
// Function to load input feature map tile block from DRAM to on-chip BRAM.
//
// TODO: This is a template function that you need to modify in Task B 
//       to handle border conditions. 
//--------------------------------------------------------------------------
void load_input_tile_block_from_DRAM_id3 (
    fm_t in_fm_buf[IN_BUF_DEPTH1][IN_BUF_HEIGHT][IN_BUF_WIDTH], 
    fm_t in_fm[1024][416][416], 
    int  ti,
    int  tj, 
    int  d
)
{
    const int depth_offset  =  d * IN_BUF_DEPTH1;
    const int height_offset = ti * OUT_BUF_HEIGHT; // OUT_BUF is intended, not a typo. 
    const int width_offset  = tj * OUT_BUF_WIDTH;
        
    INPUT_BUFFER_DEPTH:
    for(int c = 0; c < IN_BUF_DEPTH1; c++)
    {
        INPUT_BUFFER_HEIGHT:
        for(int i = 0; i < IN_BUF_HEIGHT; i++)
        {
            INPUT_BUFFER_WIDTH:
            for(int j = 0; j < IN_BUF_WIDTH; j++)
            {
                // TODO: Handle border features here
                if(((ti == 0) && (i == 0)) || ((ti == N_TILE_ROWS - 1) && (i == IN_BUF_HEIGHT - 1)))
                    in_fm_buf[c][i][j] = 0;
                else if(((tj == 0) && (j == 0)) || ((tj == N_TILE_COLS - 1) && (j == IN_BUF_WIDTH - 1)))
                    in_fm_buf[c][i][j] = 0;
                else
		            in_fm_buf[c][i][j] = in_fm[depth_offset + c][height_offset + i - 1][width_offset + j - 1];
            }
        }
    }
}

//--------------------------------------------------------------------------
// Function to load layer parameters (weights and bias) for convolution.
//
// You should not need to modify this function in Task B. 
//
// TODO: In Task C, you may add pragmas depending on your way 
//       of optimization.
//--------------------------------------------------------------------------
void load_layer_params_from_DRAM_id3 (
    wt_t weight_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH1][3][3],
    wt_t weights[1024][1024][3][3],
    int b,
    int d
)
{
    const int kernel_offset  = b * OUT_BUF_DEPTH;
    const int channel_offset = d * IN_BUF_DEPTH1;

    WEIGHT_KERNEL_NUM:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
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
	                weight_buf[f][c][kh][kw] = weights[kernel_offset + f][channel_offset + c][kh][kw];
                }
            }
        }
    }
    // BATCH_NORM:
    // for(int f = 0; f < OUT_BUF_DEPTH; f++)
    // {
    //     BATCH_NORM_PARAMS:
    //     for (int i = 0; i < 4; i++)
    //     {
    //         bn_buf[f][i] = bn[kernel_offset + f][i];
    //     }
    // }
}

//--------------------------------------------------------------------------
// Function to load input feature map tile block from DRAM to on-chip BRAM.
//
// TODO: This is a template function that you need to modify in Task B 
//       to handle border conditions. 
//--------------------------------------------------------------------------
void load_input_tile_block_from_DRAM_id16 (
    fm_t in_fm_buf[IN_BUF_DEPTH2][IN_BUF_HEIGHT][IN_BUF_WIDTH], 
    fm_t in_fm[1024][416][416], 
    int  ti,
    int  tj, 
    int  d
)
{
    const int depth_offset  =  d * IN_BUF_DEPTH2;
    const int height_offset = ti * OUT_BUF_HEIGHT; // OUT_BUF is intended, not a typo. 
    const int width_offset  = tj * OUT_BUF_WIDTH;
        
    INPUT_BUFFER_DEPTH:
    for(int c = 0; c < IN_BUF_DEPTH2; c++)
    {
        INPUT_BUFFER_HEIGHT:
        for(int i = 0; i < IN_BUF_HEIGHT; i++)
        {
            INPUT_BUFFER_WIDTH:
            for(int j = 0; j < IN_BUF_WIDTH; j++)
            {
                // TODO: Handle border features here
                if(((ti == 0) && (i == 0)) || ((ti == N_TILE_ROWS - 1) && (i == IN_BUF_HEIGHT - 1)))
                    in_fm_buf[c][i][j] = 0;
                else if(((tj == 0) && (j == 0)) || ((tj == N_TILE_COLS - 1) && (j == IN_BUF_WIDTH - 1)))
                    in_fm_buf[c][i][j] = 0;
                else
		            in_fm_buf[c][i][j] = in_fm[depth_offset + c][height_offset + i - 1][width_offset + j - 1];
            }
        }
    }
}

//--------------------------------------------------------------------------
// Function to load layer parameters (weights and bias) for convolution.
//
// You should not need to modify this function in Task B. 
//
// TODO: In Task C, you may add pragmas depending on your way 
//       of optimization.
//--------------------------------------------------------------------------
void load_layer_params_from_DRAM_id16 (
    wt_t weight_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH2][3][3],
    wt_t weights[1024][1024][3][3],
    int b,
    int d
)
{
    const int kernel_offset  = b * OUT_BUF_DEPTH;
    const int channel_offset = d * IN_BUF_DEPTH2;

    WEIGHT_KERNEL_NUM:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
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
	                weight_buf[f][c][kh][kw] = weights[kernel_offset + f][channel_offset + c][kh][kw];
                }
            }
        }
    }
    // BATCH_NORM:
    // for(int f = 0; f < OUT_BUF_DEPTH; f++)
    // {
    //     BATCH_NORM_PARAMS:
    //     for (int i = 0; i < 4; i++)
    //     {
    //         bn_buf[f][i] = bn[kernel_offset + f][i];
    //     }
    // }
}

void load_layer_params_from_DRAM_conv1x1 (
    wt_t weight_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH2][1][1],
    wt_t weights[1024][1024][1][1],
    int b,
    int d
)
{
    const int kernel_offset  = b * OUT_BUF_DEPTH;
    const int channel_offset = d * IN_BUF_DEPTH2;

    WEIGHT_KERNEL_NUM_1x1:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        WEIGHT_KERNEL_DEPTH_1x1:
        for(int c = 0; c < IN_BUF_DEPTH2; c++)
        {
            weight_buf[f][c][0][0] = weights[kernel_offset + f][channel_offset + c][0][0];
        }
    }
}

void load_layer_params_from_DRAM_bias (
    wt_t weight_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH2][1][1],
    wt_t bias_buf[OUT_BUF_DEPTH],
    wt_t weights[1024][1024][1][1],
    wt_t bias[255],
    int b,
    int d
)
{
    const int kernel_offset  = b * OUT_BUF_DEPTH;
    const int channel_offset = d * IN_BUF_DEPTH2;

    WEIGHT_KERNEL_NUM_BIAS:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        WEIGHT_KERNEL_DEPTH_BIAS:
        for(int c = 0; c < IN_BUF_DEPTH2; c++)
        {
            weight_buf[f][c][0][0] = weights[kernel_offset + f][channel_offset + c][0][0];
        }
    }

    BIAS:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        bias_buf[f] = bias[kernel_offset + f];
    }
}

//--------------------------------------------------------------------------
// Function to load input feature map tile block from DRAM to on-chip BRAM.
//
// TODO: This is a template function that you need to modify in Task B 
//       to handle border conditions. 
//--------------------------------------------------------------------------
void load_input_tile_block_from_DRAM_conv (
    fm_t in_fm_buf[IN_BUF_DEPTH2][IN_BUF_HEIGHT2][IN_BUF_WIDTH2], 
    fm_t in_fm[1024][416][416], 
    int  ti,
    int  tj, 
    int  d
)
{
    const int depth_offset  =  d * IN_BUF_DEPTH2;
    const int height_offset = ti * OUT_BUF_HEIGHT2; // OUT_BUF is intended, not a typo. 
    const int width_offset  = tj * OUT_BUF_WIDTH2;
        
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
                if(((ti == 0) && (i == 0)) || ((ti == N_TILE_ROWS - 1) && (i == IN_BUF_HEIGHT - 1)))
                    in_fm_buf[c][i][j] = 0;
                else if(((tj == 0) && (j == 0)) || ((tj == N_TILE_COLS - 1) && (j == IN_BUF_WIDTH - 1)))
                    in_fm_buf[c][i][j] = 0;
                else
		            in_fm_buf[c][i][j] = in_fm[depth_offset + c][height_offset + i - 1][width_offset + j - 1];
            }
        }
    }
}

//------------------------------------------------------------------------------
// Function to save partial outputs on-chip for each input tile block processed.
//
// You should not need to modify this function in Task B. 
//
// TODO: In Task C, you may add pragmas depending on your way 
//       of optimization.
//------------------------------------------------------------------------------
void save_partial_output_tile_block (
    fm_t partial_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH],
    int d
)
{
    //fm_t A, B, epsilon = 0.001;

    PARTIAL_OUTPUT_BUFFER_DEPTH:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        PARTIAL_OUTPUT_BUFFER_HEIGHT:
        for(int i = 0; i < OUT_BUF_HEIGHT; i++)
        {
            PARTIAL_OUTPUT_BUFFER_WIDTH:
            for(int j = 0; j < OUT_BUF_WIDTH; j++)
            {
                //A = bn_buf[f][0] / (fm_t)(sqrt(bn_buf[f][3] + epsilon));
                //B = bn_buf[f][1] - ((bn_buf[f][0]*bn_buf[f][2])/ (fm_t)(sqrt(bn_buf[f][3] + epsilon)));
                if(d == 0) // Initialize buffer for first kernel group
                {
                    partial_out_buf[f][i][j]   = out_fm_buf[f][i][j];
                }
                else // Accumulate otherwise
                {
                    partial_out_buf[f][i][j]  += out_fm_buf[f][i][j];
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// Function to save partial outputs on-chip for each input tile block processed.
//
// You should not need to modify this function in Task B. 
//
// TODO: In Task C, you may add pragmas depending on your way 
//       of optimization.
//------------------------------------------------------------------------------
void save_partial_output_tile_block_conv (
    fm_t partial_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2], 
    fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2],
    int d
)
{
    PARTIAL_OUTPUT_BUFFER_DEPTH:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        PARTIAL_OUTPUT_BUFFER_HEIGHT:
        for(int i = 0; i < OUT_BUF_HEIGHT2; i++)
        {
            PARTIAL_OUTPUT_BUFFER_WIDTH:
            for(int j = 0; j < OUT_BUF_WIDTH2; j++)
            {
                if(d == 0) // Initialize buffer for first kernel group
                {
                    partial_out_buf[f][i][j] = out_fm_buf[f][i][j];
                }
                else // Accumulate otherwise
                {
                    partial_out_buf[f][i][j] += out_fm_buf[f][i][j];
                }
            }
        }
    }
}

void save_partial_output_tile_block_bias (
    fm_t partial_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2], 
    fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2],
    wt_t bias_buf[OUT_BUF_DEPTH],
    int d
)
{
    PARTIAL_OUTPUT_BUFFER_DEPTH_BIAS:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        PARTIAL_OUTPUT_BUFFER_HEIGHT_BIAS:
        for(int i = 0; i < OUT_BUF_HEIGHT2; i++)
        {
            PARTIAL_OUTPUT_BUFFER_WIDTH_BIAS:
            for(int j = 0; j < OUT_BUF_WIDTH2; j++)
            {
                if(d == 0) // Initialize buffer for first kernel group
                {
                    partial_out_buf[f][i][j]   = out_fm_buf[f][i][j] + bias_buf[f];
                }
                else // Accumulate otherwise
                {
                    partial_out_buf[f][i][j]  += out_fm_buf[f][i][j];
                }
            }
        }
    }
}

//--------------------------------------------------------------------------
// Function to store complete output tile block from BRAM to DRAM.
//
// You should not need to modify this function. 
//--------------------------------------------------------------------------
void store_output_tile_to_DRAM (
    fm_t out_fm[1024][416][416], 
    fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2], 
    int  ti,
    int  tj,
    int  b
)
{
    const int depth_offset  =  b * OUT_BUF_DEPTH;
    const int height_offset = ti * OUT_BUF_HEIGHT2;
    const int width_offset  = tj * OUT_BUF_WIDTH2;

    OUTPUT_BUFFER_DEPTH:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        OUTPUT_BUFFER_HEIGHT:
        for(int i = 0; i < OUT_BUF_HEIGHT2; i++)
        {
            OUTPUT_BUFFER_WIDTH:
            for(int j = 0; j < OUT_BUF_WIDTH2; j++)
            {
                // Leaky ReLU in-place
                if(out_fm_buf[f][i][j] < (fm_t) 0)
                {
                    out_fm[depth_offset + f][height_offset + i][width_offset + j] = (fm_t)(0.1) * out_fm_buf[f][i][j];
                }
                else
                {
                    out_fm[depth_offset + f][height_offset + i][width_offset + j] = out_fm_buf[f][i][j];
                }
            }
        }
    }
}

void store_output_tile_to_DRAM_ver1 (
    fm_t out_fm[1024][416][416], 
    fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH], 
    int  ti,
    int  tj,
    int  b
)
{
    const int depth_offset  =  b * OUT_BUF_DEPTH;
    const int height_offset = ti * OUT_BUF_HEIGHT;
    const int width_offset  = tj * OUT_BUF_WIDTH;

    OUTPUT_BUFFER_DEPTH_VER1:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        OUTPUT_BUFFER_HEIGHT_VER1:
        for(int i = 0; i < OUT_BUF_HEIGHT; i++)
        {
            OUTPUT_BUFFER_WIDTH_VER1:
            for(int j = 0; j < OUT_BUF_WIDTH; j++)
            {
                // Leaky ReLU in-place
                if(out_fm_buf[f][i][j] < (fm_t) 0)
                {
                    out_fm[depth_offset + f][height_offset + i][width_offset + j] = (fm_t)(0.1) * out_fm_buf[f][i][j];
                }
                else
                {
                    out_fm[depth_offset + f][height_offset + i][width_offset + j] = out_fm_buf[f][i][j];
                }
            }
        }
    }
}

void store_maxpool_output_tile_to_DRAM_stride1(
    fm_t out_fm[1024][416][416], 
    fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT2][OUT_BUF_WIDTH2],
    int d
)
{
    const int depth_offset  =  d * OUT_BUF_DEPTH;

    MAXPOOL_OUTPUT_BUFFER_DEPTH:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        MAXPOOL_OUTPUT_BUFFER_HEIGHT:
        for(int i = 0; i < OUT_BUF_HEIGHT2; i++)
        {
            MAXPOOL_OUTPUT_BUFFER_WIDTH:
            for(int j = 0; j < OUT_BUF_WIDTH2; j++)
            {
                out_fm[depth_offset + f][i][j] = out_fm_buf[f][i][j];
            }
        }
    }
}

void store_upsample_output_tile_to_DRAM(
    fm_t out_fm[1024][416][416],
    fm_t out_fm_buf[US_BUF_DEPTH][US_BUF_HEIGHT*2][US_BUF_WIDTH*2],
    int d
)
{
    const int depth_offset  =  d * US_BUF_DEPTH;

    UPSAMPLE_OUTPUT_BUFFER_DEPTH:
    for(int f = 0; f < US_BUF_DEPTH; f++)
    {
        UPSAMPLE_OUTPUT_BUFFER_HEIGHT:
        for(int i = 0; i < US_BUF_HEIGHT*2; i++)
        {
            UPSAMPLE_OUTPUT_BUFFER_WIDTH:
            for(int j = 0; j < US_BUF_WIDTH*2; j++)
            {
                out_fm[depth_offset + f][i][j] = out_fm_buf[f][i][j];
            }
        }
    }
}

//--------------------------------------------------------------------------
// Function to store maxpool output tile block from BRAM to DRAM.
//
// You should not need to modify this function. 
//--------------------------------------------------------------------------
void store_maxpool_output_tile_to_DRAM (
    fm_t out_fm[1024][416][416], 
    fm_t out_fm_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT/2][OUT_BUF_WIDTH/2], 
    int  ti,
    int  tj,
    int  b
)
{
    const int depth_offset  =  b * OUT_BUF_DEPTH;
    const int height_offset = ti * (OUT_BUF_HEIGHT/2);
    const int width_offset  = tj * (OUT_BUF_WIDTH/2);

    OUTPUT_BUFFER_DEPTH:
    for(int f = 0; f < OUT_BUF_DEPTH; f++)
    {
        OUTPUT_BUFFER_HEIGHT:
        for(int i = 0; i < (OUT_BUF_HEIGHT/2); i++)
        {
            OUTPUT_BUFFER_WIDTH:
            for(int j = 0; j < (OUT_BUF_WIDTH/2); j++)
            {
                // Leaky ReLU in-place
                if(out_fm_buf[f][i][j] < (fm_t) 0)
                {
                    out_fm[depth_offset + f][height_offset + i][width_offset + j] = (fm_t)(0.1) * out_fm_buf[f][i][j];
                }
                else
                {
                    out_fm[depth_offset + f][height_offset + i][width_offset + j] = out_fm_buf[f][i][j];
                }
            }
        }
    }
}
