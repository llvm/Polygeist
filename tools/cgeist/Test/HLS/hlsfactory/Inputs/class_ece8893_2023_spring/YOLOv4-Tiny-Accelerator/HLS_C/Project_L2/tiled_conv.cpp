///////////////////////////////////////////////////////////////////////////////
// Author:      <>
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    tiled_conv.cpp
// Description: Implement a synthesizable tiling-based convolution for
//              ResNet-50's first 7x7 layer with an HD input image.
//
// TODO: Use your unoptimized code from Part B and apply your favorite pragmas
//       to achieve the target speedup (or higher)!
//
//       Add/remove/modify whatever you like.
///////////////////////////////////////////////////////////////////////////////
#include "utils.h"
#define num_kg_grps OUT_FM_DEPTH / OUT_BUF_DEPTH
void tiled_conv(
    fm_t input_feature_map[IN_FM_DEPTH][IN_FM_HEIGHT][IN_FM_WIDTH],
    wt_t layer_weights[OUT_FM_DEPTH][IN_FM_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH],
    wt_t layer_bias[OUT_FM_DEPTH],
    fm_t output_feature_map[OUT_FM_DEPTH][OUT_FM_HEIGHT][OUT_FM_WIDTH])
{
    //--------------------------------------------------------------------------
    // Defines interface IO ports for HLS.
    //--------------------------------------------------------------------------

#pragma HLS INTERFACE m_axi depth = 1 port = input_feature_map bundle = fm
#pragma HLS INTERFACE m_axi depth = 1 port = layer_weights bundle = wt
#pragma HLS INTERFACE m_axi depth = 1 port = layer_bias bundle = wt
#pragma HLS INTERFACE m_axi depth = 1 port = output_feature_map bundle = fm

#pragma HLS INTERFACE s_axilite register port = return

    //--------------------------------------------------------------------------
    // On-chip buffers
    // You should NOT modify the buffer dimensions!
    //--------------------------------------------------------------------------
    /*
    fm_t conv_in_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH];
    wt_t conv_wt_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH];
    wt_t conv_bias_buf[OUT_BUF_DEPTH];
    fm_t conv_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH] = {0};
*/

    fm_t conv_in_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH];
    wt_t conv_wt_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH];
    wt_t conv_bias_buf[OUT_BUF_DEPTH];
    fm_t conv_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH]; //= {0};

//--------------------------------------------------------------------------
// Process each tile iteratively
//--------------------------------------------------------------------------
TILE_ROW:
    for (int ti = 0; ti < N_TILE_ROWS; ti++)
    {

    TILE_COL:
        for (int tj = 0; tj < N_TILE_COLS; tj++)
        {
            load_input_tile_block_from_DRAM(conv_in_buf, input_feature_map, ti, tj);

        KERNEL_GRP:
            for (int kg = 0; kg < num_kg_grps; kg++)
            {

                // std::cout << "Processing Tile " << ti * N_TILE_COLS + tj + 1;
                // std::cout << "/" << N_TILE_ROWS * N_TILE_COLS << std::endl;

                //--------------------------------------------------------------------------
                // TODO: Your code for Part C goes here
                //
                // You may use your Part B code here and make any modifications
                // to optimize your design.
                //--------------------------------------------------------------------------

                load_layer_params_from_DRAM(conv_wt_buf, conv_bias_buf, layer_weights, layer_bias, kg);

                conv_7x7(conv_out_buf, conv_in_buf, conv_wt_buf, conv_bias_buf);

                store_output_tile_to_DRAM(output_feature_map, conv_out_buf, ti, tj, kg);
            }
        }
    }
}