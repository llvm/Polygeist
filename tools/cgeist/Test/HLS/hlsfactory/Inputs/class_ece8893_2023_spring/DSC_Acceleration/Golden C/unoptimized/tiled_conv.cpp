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

void tiled_conv (
    fm_t input_feature_map[IN_FM_DEPTH][IN_FM_HEIGHT][IN_FM_WIDTH],

    fm_t fixp_conv_ifmap_0   [32][48][48],
    fm_t fixp_conv_ifmap_1_0 [32][48][48],
    fm_t fixp_conv_ifmap_2_0 [16][48][48],
    fm_t fixp_conv_ifmap_2_1 [96][48][48],
    fm_t fixp_conv_ifmap_2_2 [96][24][24],
 
    wt_t layer_weights_0   [32][3][3][3],
    wt_t layer_bias_0      [32],
    wt_t layer_weights_1_0 [32][1][3][3],
    wt_t layer_bias_1_0    [32],
    wt_t layer_weights_1_1 [16][32][1][1],
    wt_t layer_bias_1_1    [16],
    wt_t layer_weights_2_0 [96][16][1][1],
    wt_t layer_bias_2_0    [96],
    wt_t layer_weights_2_1 [1][96][3][3],
    wt_t layer_bias_2_1    [96],
    wt_t layer_weights_2_2 [24][96][1][1],
    wt_t layer_bias_2_2    [24],

    fm_t output_feature_map[24][24][24]
)
{
    //--------------------------------------------------------------------------
    // Defines interface IO ports for HLS. 
    //--------------------------------------------------------------------------
    #pragma HLS INTERFACE m_axi depth=3*96*96   port=input_feature_map   bundle=fm

    #pragma HLS INTERFACE m_axi depth=32*48*48   port=fixp_conv_ifmap_0     bundle=fm
    #pragma HLS INTERFACE m_axi depth=32*48*48   port=fixp_conv_ifmap_1_0   bundle=fm
    #pragma HLS INTERFACE m_axi depth=16*48*48   port=fixp_conv_ifmap_2_0   bundle=fm
    #pragma HLS INTERFACE m_axi depth=96*48*48   port=fixp_conv_ifmap_2_1   bundle=fm
    #pragma HLS INTERFACE m_axi depth=96*24*24   port=fixp_conv_ifmap_2_2   bundle=fm

    #pragma HLS INTERFACE m_axi depth=32*3*3*3  port=layer_weights_0     bundle=wt
    #pragma HLS INTERFACE m_axi depth=32*32*3*3 port=layer_weights_1_0   bundle=wt
    #pragma HLS INTERFACE m_axi depth=16*32*3*3 port=layer_weights_1_1   bundle=wt
    #pragma HLS INTERFACE m_axi depth=96*16*1*1 port=layer_weights_2_0   bundle=wt
    #pragma HLS INTERFACE m_axi depth=96*96*3*3 port=layer_weights_2_1   bundle=wt
    #pragma HLS INTERFACE m_axi depth=24*96*1*1 port=layer_weights_2_2   bundle=wt
 
    #pragma HLS INTERFACE m_axi depth=32        port=layer_bias_0        bundle=wt
    #pragma HLS INTERFACE m_axi depth=32        port=layer_bias_1_0      bundle=wt
    #pragma HLS INTERFACE m_axi depth=16        port=layer_bias_1_1      bundle=wt
    #pragma HLS INTERFACE m_axi depth=96        port=layer_bias_2_0      bundle=wt
    #pragma HLS INTERFACE m_axi depth=96        port=layer_bias_2_1      bundle=wt
    #pragma HLS INTERFACE m_axi depth=24        port=layer_bias_2_2      bundle=wt
 
    #pragma HLS INTERFACE m_axi depth=24*24*24  port=output_feature_map  bundle=fm
    
    #pragma HLS INTERFACE s_axilite register        port=return
    
    //--------------------------------------------------------------------------
    // On-chip buffers
    // You should NOT modify the buffer dimensions!
    //--------------------------------------------------------------------------
    fm_t conv_in_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH];
    wt_t conv_wt_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH];
    wt_t conv_bias_buf[OUT_BUF_DEPTH];
    fm_t conv_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH] = {0};
    
    //--------------------------------------------------------------------------
    // Process each tile iteratively
    //--------------------------------------------------------------------------

    // #pragma HLS array_partition variable=conv_out_buf dim=3  complete
    // #pragma HLS array_partition variable=conv_in_buf  dim=3  complete
    #pragma HLS inline off

    ////////////////////////////////////////////////////////////////////////
    //  LAYER 1 - conv2D stride 2
    ////////////////////////////////////////////////////////////////////////
    TILE_ROW_1:
    for(int ti = 0; ti < N_TILE_ROWS; ti++)
    {
        TILE_COL_1:
        for(int tj = 0; tj < N_TILE_COLS; tj++)
        {
            std::cout << "Processing Tile " << ti*N_TILE_COLS + tj + 1;
            std::cout << "/" << N_TILE_ROWS * N_TILE_COLS << std::endl;

            KERNEL_OUTPUT_CHANNELS_1:
            for(int ki = 0; ki <= (int) 32/(OUT_BUF_DEPTH + 1); ki++) // adding 1 instead of using ceil function
            {
                KERNEL_INPUT_CHANNELS_1:
                for(int kj = 0; kj <= (int) 3/(IN_BUF_DEPTH + 1); kj++)
                {
                    load_input_tile_block_from_DRAM <3, 96, 96>(conv_in_buf, input_feature_map, ti, tj, kj, 1);
                    load_conv_layer_params_from_DRAM <32, 3, 3, 3>(conv_wt_buf, conv_bias_buf, layer_weights_0, layer_bias_0, ki, kj);
                    conv_3x3 (conv_out_buf, conv_in_buf, conv_wt_buf, conv_bias_buf, ki, kj, 2, 3, 32);
                }
                reLU6_1 (conv_out_buf);
                store_output_tile_to_DRAM <32, 48, 48>(fixp_conv_ifmap_0, conv_out_buf, ti, tj, ki, 2);
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////
    //  LAYER 2 - Depthwise Separable Convolution Stride 1
    ////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////
    // Layer 2_1 - Depthwise Convolution 
    ////////////////////////////////////////////
    TILE_ROW_2_1:
    for(int ti = 0; ti <= (int) 48 / TILE_HEIGHT; ti++)
    {
        TILE_COL_2_1:
        for(int tj = 0; tj <= (int) 48  / TILE_WIDTH; tj++)
        {
            std::cout << "[LAYER1] Processing Tile " << (ti * 2) + tj + 1 << std::endl;
            KERNEL_OUTPUT_CHANNELS_2_1:
            for(int kj = 0; kj <= (int) 32/(IN_BUF_DEPTH + 1); kj++)
            {
                load_input_tile_block_from_DRAM <32, 48, 48>(conv_in_buf, fixp_conv_ifmap_0, ti, tj, kj, 1);
                load_conv_layer_params_from_DRAM <32, 1, 3, 3>(conv_wt_buf, conv_bias_buf, layer_weights_1_0, layer_bias_1_0, kj, 0);//TODO optimized by only getting 1 output channel instead of 32
                dconv_3x3 (conv_out_buf, conv_in_buf, conv_wt_buf, conv_bias_buf, kj, 1, 32);
                reLU6_1 (conv_out_buf);
                store_output_tile_to_DRAM <32, 48, 48>(fixp_conv_ifmap_1_0, conv_out_buf, ti, tj, kj, 1);
            }
        }
    }

    /////////////////////////////////////////////
    // Layer 2_2 - Pointwise Convolution 
    ////////////////////////////////////////////
    TILE_ROW_2_2:
    for(int ti = 0; ti <= (int) 48 / TILE_HEIGHT; ti++)
    {
        TILE_COL_2_2:
        for(int tj = 0; tj <= (int) 48  / TILE_WIDTH; tj++)
        {
            std::cout << "[LAYER1_2] Processing Tile " << (ti * 2) + tj + 1 << std::endl;

            KERNEL_OUTPUT_CHANNELS_2_2:
            for(int ki = 0; ki <= (int) 16/(OUT_BUF_DEPTH + 1); ki++)
            {
                KERNEL_INPUT_CHANNELS_2_2:
                for(int kj = 0; kj <= (int) 32/(IN_BUF_DEPTH + 1); kj++)
                {
                    load_input_tile_block_from_DRAM <32, 48, 48>(conv_in_buf, fixp_conv_ifmap_1_0, ti, tj, kj, 0);
                    load_conv_layer_params_from_DRAM <16, 32, 1, 1>(conv_wt_buf, conv_bias_buf, layer_weights_1_1, layer_bias_1_1, ki, kj);
                    pointwise_conv_3x3 (conv_out_buf, conv_in_buf, conv_wt_buf, conv_bias_buf, ki, kj, 32, 16);
                }
                store_output_tile_to_DRAM <16, 48, 48>(fixp_conv_ifmap_2_0, conv_out_buf, ti, tj, ki, 1);
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////
    //  LAYER 3 - Depthwise Separable Convolution Stride 2
    ////////////////////////////////////////////////////////////////////////

    /////////////////////////////////////////////
    // Layer 3_1 - Pointwise Convolution 
    ////////////////////////////////////////////
    TILE_ROW_3_1:
    for(int ti = 0; ti <= (int) 48 / TILE_HEIGHT; ti++)
    {
        TILE_COL_3_1:
        for(int tj = 0; tj <= (int) 48  / TILE_WIDTH; tj++)
        {
            std::cout << "[LAYER3_1] Processing Tile " << (ti * 2) + tj + 1 << std::endl;

            KERNEL_OUTPUT_CHANNELS_3_1:
            for(int ki = 0; ki <= (int) 96/(OUT_BUF_DEPTH + 1); ki++)
            {
                KERNEL_INPUT_CHANNELS_3_1:
                for(int kj = 0; kj <= (int) 16/(IN_BUF_DEPTH + 1); kj++)
                {
                    load_input_tile_block_from_DRAM <16, 48, 48>(conv_in_buf, fixp_conv_ifmap_2_0, ti, tj, kj, 0);
                    load_conv_layer_params_from_DRAM <96, 16, 1, 1>(conv_wt_buf, conv_bias_buf, layer_weights_2_0, layer_bias_2_0, ki, kj);
                    pointwise_conv_3x3 (conv_out_buf, conv_in_buf, conv_wt_buf, conv_bias_buf, ki, kj, 16, 96);
                }
                reLU6_1 (conv_out_buf);
                store_output_tile_to_DRAM <96, 48, 48>(fixp_conv_ifmap_2_1, conv_out_buf, ti, tj, ki, 1);
            }
        }
    }

    /////////////////////////////////////////////
    // Layer 3_2 - Depthwise Convolution 
    ////////////////////////////////////////////
    TILE_ROW_3_2:
    for(int ti = 0; ti <= (int) 48 / TILE_HEIGHT; ti++)
    {
        TILE_COL_3_2:
        for(int tj = 0; tj <= (int) 48  / TILE_WIDTH; tj++)
        {
            std::cout << "[LAYER3_2] Processing Tile " << (ti * 2) + tj + 1 << std::endl;
            KERNEL_OUTPUT_CHANNELS_3_2:
            for(int kj = 0; kj <= (int) 96/(IN_BUF_DEPTH + 1); kj++)
            {
                load_input_tile_block_from_DRAM <96, 48, 48>(conv_in_buf, fixp_conv_ifmap_2_1, ti, tj, kj, 1);
                load_conv_layer_params_from_DRAM <96, 96, 3, 3>(conv_wt_buf, conv_bias_buf, layer_weights_2_1, layer_bias_2_1, 0, kj);//TODO optimized by only getting 1 output channel instead of 32
                d2conv_3x3 (conv_out_buf, conv_in_buf, conv_wt_buf, conv_bias_buf, kj, 2, 96);
                reLU6_2 (conv_out_buf);
                store_output_tile_to_DRAM <96, 24, 24>(fixp_conv_ifmap_2_2, conv_out_buf, ti, tj, kj, 2);
            }
        }
    }

    /////////////////////////////////////////////
    // Layer 3_3 - Pointwise Convolution 
    ////////////////////////////////////////////
    TILE_ROW_3_3:
    for(int ti = 0; ti <= (int) 24 / TILE_HEIGHT; ti++)
    {
        TILE_COL_3_3:
        for(int tj = 0; tj <= (int) 24  / TILE_WIDTH; tj++)
        {
            std::cout << "[LAYER3_3] Processing Tile " << (ti * 2) + tj + 1 << std::endl;

            KERNEL_OUTPUT_CHANNELS_3_3:
            for(int ki = 0; ki <= (int) 24/(OUT_BUF_DEPTH + 1); ki++)
            {
                KERNEL_INPUT_CHANNELS_3_3:
                for(int kj = 0; kj <= (int) 96/(IN_BUF_DEPTH + 1); kj++)
                {
                    load_input_tile_block_from_DRAM <96, 24, 24>(conv_in_buf, fixp_conv_ifmap_2_2, ti, tj, kj, 0);
                    load_conv_layer_params_from_DRAM <24, 96, 1, 1>(conv_wt_buf, conv_bias_buf, layer_weights_2_2, layer_bias_2_2, ki, kj);
                    pointwise_conv_3x3 (conv_out_buf, conv_in_buf, conv_wt_buf, conv_bias_buf, ki, kj, 96, 24);
                }
                store_output_tile_to_DRAM <24, 24, 24>(output_feature_map, conv_out_buf, ti, tj, ki, 1);
            }
        }
    }
}
