///////////////////////////////////////////////////////////////////////////////
// Course:      ECE8893 - Parallel Programming for FPGAs
// Filename:    tiled_conv.cpp
// Description: Implement a synthesizable tiling-based convolution for
//              ResNet-50's first 7x7 layer with an HD input image.
//
// TODO: Partition arrays appropriately to accelerate computation!
///////////////////////////////////////////////////////////////////////////////
#include "utils.h"

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
    fm_t conv_in_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH];
    wt_t conv_wt_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH];
    wt_t conv_bias_buf[OUT_BUF_DEPTH];
    fm_t conv_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH] = {0};

//--------------------------------------------------------------------------
// Process each tile iteratively
//--------------------------------------------------------------------------
TILE_ROW:
    for (int ti = 0; ti < N_TILE_ROWS; ti++)
    {
    TILE_COL:
        for (int tj = 0; tj < N_TILE_COLS; tj++)
        {
            // std::cout << "Processing Tile " << ti*N_TILE_COLS + tj + 1;
            //  std::cout << "/" << N_TILE_ROWS * N_TILE_COLS << std::endl;

            // Load input tile block
            load_input_tile_block_from_DRAM(conv_in_buf,
                                            input_feature_map,
                                            ti,
                                            tj);
        FILTER_SIZE:
            for (int kg = 0; kg < (OUT_FM_DEPTH / OUT_BUF_DEPTH); kg++)
            {
                // Load weights and bias groups from DRAM
                load_layer_params_from_DRAM(conv_wt_buf,
                                            conv_bias_buf,
                                            layer_weights,
                                            layer_bias,
                                            kg);

                // Run 7x7 convolution on loaded tile
                conv_7x7(conv_out_buf,
                         conv_in_buf,
                         conv_wt_buf,
                         conv_bias_buf);

                // Store final output tile to DRAM
                store_output_tile_to_DRAM(output_feature_map,
                                          conv_out_buf,
                                          ti,
                                          tj,
                                          kg);
            }
        }
    }
}
