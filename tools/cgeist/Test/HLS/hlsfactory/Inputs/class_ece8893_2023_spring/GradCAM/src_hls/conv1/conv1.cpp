#include "../util.h"
#include <iostream>
#include <cassert>

namespace conv1
{

#include "params.hpp"
#include "conv.cpp"
#include "io.cpp"

static_assert(STRIDE == 1 || STRIDE == 2, "STRIDE > 2 not implemented");
static_assert(TILE_HEIGHT % STRIDE == 0, "TILE_HEIGHT must be a multiple of STRIDE");
static_assert(TILE_WIDTH % STRIDE == 0, "TILE_WIDTH must be a multiple of STRIDE");
static_assert(IN_FM_HEIGHT % STRIDE == 0);
static_assert(IN_FM_WIDTH % STRIDE == 0);
static_assert(OUT_FM_DEPTH % OUT_BUF_DEPTH == 0);
static_assert(IN_FM_HEIGHT % TILE_HEIGHT == 0);
static_assert(IN_FM_WIDTH % TILE_WIDTH == 0);

void tiled_conv(
    fm_t output_feature_map[64][112][112],
    const fm_t input_feature_map[3][224][224],
    const wt_t layer_weights[64][3][7][7],
    const wt_t layer_bias[64]
)
{
    #pragma HLS INTERFACE m_axi depth=1  port=input_feature_map   bundle=fm_in
    #pragma HLS INTERFACE m_axi depth=1  port=layer_weights       bundle=wt
    #pragma HLS INTERFACE m_axi depth=1  port=layer_bias          bundle=wt
    #pragma HLS INTERFACE m_axi depth=1  port=output_feature_map  bundle=fm_out
    #pragma HLS INTERFACE s_axilite register	port=return

    static fm_t conv_in_buf[IN_BUF_DEPTH][IN_BUF_HEIGHT][IN_BUF_WIDTH];
    static wt_t conv_wt_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH][KERNEL_HEIGHT][KERNEL_WIDTH];
    static wt_t conv_bias_buf[OUT_BUF_DEPTH];
    static fm_t conv_out_buf[OUT_BUF_DEPTH][OUT_BUF_HEIGHT][OUT_BUF_WIDTH];

    TILE_ROW:
    for(int ti = 0; ti < N_TILE_ROWS; ti++)
    {
        TILE_COL:
        for(int tj = 0; tj < N_TILE_COLS; tj++)
        {

            conv1::load_fm_tile_block_from_DRAM
                <IN_BUF_DEPTH, TILE_HEIGHT, TILE_WIDTH, PADDING, IN_FM_DEPTH, IN_FM_HEIGHT, IN_FM_WIDTH>
                (conv_in_buf, input_feature_map, ti, tj, 0);

            KERNEL_GRP:
            for (int tk = 0; tk < KERNEL_GRPS; tk++)
            {
                conv1::load_layer_params_from_DRAM(conv_wt_buf, conv_bias_buf, layer_weights, layer_bias, tk);
                conv1::conv_small(conv_out_buf, conv_in_buf, conv_wt_buf, conv_bias_buf);

                conv1::store_output_tile_to_DRAM
                    <OUT_BUF_DEPTH, OUT_BUF_HEIGHT, OUT_BUF_WIDTH, OUT_FM_DEPTH, OUT_FM_HEIGHT, OUT_FM_WIDTH>
                    (output_feature_map, conv_out_buf, ti, tj, tk, relu);

            }
        }
    }
}

}
