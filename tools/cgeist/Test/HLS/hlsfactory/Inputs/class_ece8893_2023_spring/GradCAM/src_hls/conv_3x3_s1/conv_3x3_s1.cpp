#include "../util.h"
#include <iostream>
#include <cassert>
#include "conv_3x3_s1.hpp"
#include "io.cpp"

namespace conv_3x3_s1 {

#include "params.hpp"
#include "conv.cpp"

void tiled_conv_core (
    fm_t output_feature_map[],
    const fm_t input_feature_map[],
    const wt_t layer_weights[],
    const wt_t layer_bias[],
    const int KERNEL_GRPS,
    const int IN_FM_DEPTH,
    const int N_TILE_LAYERS,
    const int N_TILE_ROWS,
    const int N_TILE_COLS,
    const bool stride_2,
    const bool inplace_residual
)
{
    #pragma HLS inline off
    #pragma HLS INTERFACE m_axi depth=1  port=input_feature_map   bundle=fm_in
    #pragma HLS INTERFACE m_axi depth=1  port=layer_weights       bundle=wt
    #pragma HLS INTERFACE m_axi depth=1  port=layer_bias          bundle=wt
    #pragma HLS INTERFACE m_axi depth=1  port=output_feature_map  bundle=fm_out
    #pragma HLS INTERFACE s_axilite register	port=return

    static_assert(STRIDE == 1 || STRIDE == 2, "STRIDE > 2 not implemented");
    static_assert(TILE_HEIGHT % STRIDE == 0, "TILE_HEIGHT must be a multiple of STRIDE");
    static_assert(TILE_WIDTH % STRIDE == 0, "TILE_WIDTH must be a multiple of STRIDE");

    //const int IN_FM_DEPTH = IN_BUF_DEPTH * N_TILE_LAYERS;
    const int IN_FM_HEIGHT = TILE_HEIGHT * N_TILE_ROWS;
    const int IN_FM_WIDTH = TILE_WIDTH * N_TILE_COLS;
    const int OUT_FM_DEPTH = OUT_BUF_DEPTH * KERNEL_GRPS;
    const int OUT_FM_HEIGHT = stride_2 ? IN_FM_HEIGHT/2 : IN_FM_HEIGHT;
    const int OUT_FM_WIDTH = stride_2 ? IN_FM_WIDTH/2 : IN_FM_WIDTH;


    assert(IN_FM_HEIGHT % STRIDE == 0);
    assert(IN_FM_WIDTH % STRIDE == 0);
    assert(OUT_FM_DEPTH % OUT_BUF_DEPTH == 0);
    assert(IN_FM_HEIGHT % TILE_HEIGHT == 0);
    assert(IN_FM_WIDTH % TILE_WIDTH == 0);


    const fm_dims_s in_fm_dims = {IN_FM_DEPTH, IN_FM_HEIGHT, IN_FM_WIDTH};
    const fm_dims_s out_fm_dims = {OUT_FM_DEPTH, OUT_FM_HEIGHT, OUT_FM_WIDTH};

    //--------------------------------------------------------------------------
    // On-chip buffers
    //--------------------------------------------------------------------------
    static fm_t conv_in_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH];
    static wt_t conv_wt_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH];
    static wt_t conv_bias_buf[OUT_BUF_DEPTH];
    static fm_t conv_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH];

    //--------------------------------------------------------------------------
    // Process each tile iteratively
    //--------------------------------------------------------------------------
    
    TILE_ROW:
    for(int ti = 0; ti < (stride_2 ? N_TILE_ROWS/2 : N_TILE_ROWS); ti++)
    {
        TILE_COL:
        for(int tj = 0; tj < (stride_2 ? N_TILE_COLS/2 : N_TILE_COLS); tj++)
        {

            TILE_LYR:
            for (int tl = 0; tl < N_TILE_LAYERS; tl++)
            {
                int DEPTH_CHECK = IN_BUF_DEPTH < IN_FM_DEPTH ? IN_BUF_DEPTH : IN_FM_DEPTH;
                conv_3x3_s1::load_fm_tile_block_from_DRAM
                    <IN_BUF_DEPTH, IN_BUF_HEIGHT, IN_BUF_WIDTH, TILE_HEIGHT, TILE_WIDTH, PADDING>
                    (conv_in_buf, input_feature_map, 
                     DEPTH_CHECK, IN_FM_HEIGHT, IN_FM_WIDTH, ti, tj, tl, stride_2);


                KERNEL_GRP:
                for (int tk = 0; tk < KERNEL_GRPS; tk++)
                {
                    bool res = inplace_residual || (tl != 0);
                    if (res)
                    {
                        conv_3x3_s1::load_fm_tile_block_from_DRAM
                            <OUT_BUF_DEPTH, OUT_BUF_HEIGHT, OUT_BUF_WIDTH, OUT_BUF_HEIGHT, OUT_BUF_WIDTH, 0>
                            (conv_out_buf, output_feature_map, 
                             OUT_BUF_DEPTH, OUT_FM_HEIGHT, OUT_FM_WIDTH, ti, tj, tk, false);
                    }

                    conv_3x3_s1::load_layer_params_from_DRAM
                        (conv_wt_buf, conv_bias_buf, (fm_t *) layer_weights, layer_bias, 
                         OUT_FM_DEPTH, IN_FM_DEPTH, tk, tl);

                    bool add_bias = tl == 0;

                    conv_3x3_s1::conv_small(
                        conv_out_buf, conv_in_buf, conv_wt_buf, conv_bias_buf, 
                                            IN_FM_DEPTH, stride_2, res, add_bias);

                    bool relu = tl == N_TILE_LAYERS - 1;
                    conv_3x3_s1::store_output_tile_to_DRAM
                        <OUT_BUF_DEPTH, OUT_BUF_HEIGHT, OUT_BUF_WIDTH>
                        (output_feature_map, conv_out_buf, out_fm_dims, ti, tj, tk, relu);
                }

            }
        }
    }
}



void test_conv(
    fm_t output_feature_map[],
    const fm_t input_feature_map[],
    const wt_t layer_weights[],
    const wt_t layer_bias[]
)
{
    conv_3x3_s1::tiled_conv<64, 64, 56, 56>(
        output_feature_map,
        input_feature_map,
        layer_weights,
        layer_bias,
        false
        );

    conv_3x3_s1::tiled_conv<64, 64, 56, 56>(
        output_feature_map,
        input_feature_map,
        layer_weights,
        layer_bias,
        true
        );
}
        
}
