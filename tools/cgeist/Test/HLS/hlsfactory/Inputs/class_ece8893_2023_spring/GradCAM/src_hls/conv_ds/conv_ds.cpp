#include "../util.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include "conv_ds.hpp"

namespace conv_ds
{

void conv_small(
    fm_t out_buf[OUT_BUF_DEPTH][BUF_HEIGHT][BUF_WIDTH],
    const fm_t in_buf[IN_BUF_DEPTH][BUF_HEIGHT][BUF_WIDTH],
    const wt_t wt_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH],
    const wt_t bias_buf[OUT_BUF_DEPTH],
    const int tl
)
{
    #pragma HLS inline off

    #pragma HLS array_partition variable=out_buf complete dim=1
    #pragma HLS array_partition variable=in_buf complete dim=1
    #pragma HLS array_partition variable=wt_buf complete dim=1
    #pragma HLS array_partition variable=bias_buf complete dim=1

CONV_IN_D: for (int c = 0; c < IN_BUF_DEPTH; c++)
    CONV_ROW: for (int i = 0; i < BUF_HEIGHT; i++)
        CONV_COL: for (int j = 0; j < BUF_WIDTH; j++)
                    #pragma HLS pipeline II=1
            CONV_OUT_D: for (int f = 0; f < OUT_BUF_DEPTH; f++)
                {
                    #pragma HLS unroll
                    fm_t x = out_buf[f][i][j];
                    if (c == 0 && tl == 0)
                        x = bias_buf[f] + in_buf[c][i][j] * wt_buf[f][c];
                    else
                        x += in_buf[c][i][j] * wt_buf[f][c];

                    out_buf[f][i][j] = x;
                }
}

int index_calc(int idx_d, int idx_h, int idx_w, int IN_FM_HEIGHT, int IN_FM_WIDTH)
{
    #pragma HLS inline off
    return idx_d*IN_FM_HEIGHT*IN_FM_WIDTH + idx_h*IN_FM_WIDTH + idx_w;
}

void tiled_conv_ds_core(
    fm_t out_feature_map[],
    const fm_t in_feature_map[],
    const wt_t layer_weights[],
    const wt_t layer_bias[],
    const int N_TILE_ROWS,
    const int N_TILE_COLS,
    const int N_TILE_LAYERS,
    const int KERNEL_GROUPS
)
{
    #pragma HLS inline off

    #pragma HLS INTERFACE m_axi depth=1  port=in_feature_map   bundle=ds
    #pragma HLS INTERFACE m_axi depth=1  port=layer_weights    bundle=ds
    #pragma HLS INTERFACE m_axi depth=1  port=layer_bias       bundle=ds
    #pragma HLS INTERFACE m_axi depth=1  port=out_feature_map  bundle=ds
    #pragma HLS INTERFACE s_axilite register	port=return

    const int IN_FM_DEPTH = IN_BUF_DEPTH * N_TILE_LAYERS;
    const int IN_FM_WIDTH = 2 * BUF_WIDTH * N_TILE_COLS;
    const int IN_FM_HEIGHT = 2 * BUF_HEIGHT * N_TILE_ROWS;
    const int OUT_FM_DEPTH = OUT_BUF_DEPTH * KERNEL_GROUPS;
    const int OUT_FM_HEIGHT = IN_FM_HEIGHT / 2;
    const int OUT_FM_WIDTH = IN_FM_WIDTH / 2;

    static fm_t in_buf[IN_BUF_DEPTH][BUF_HEIGHT][BUF_WIDTH];
    static fm_t out_buf[OUT_BUF_DEPTH][BUF_HEIGHT][BUF_WIDTH];
    static fm_t wt_buf[OUT_BUF_DEPTH][IN_BUF_DEPTH];
    static fm_t bias_buf[OUT_BUF_DEPTH];

    TILE_ROW:
    for(int ti = 0; ti < N_TILE_ROWS; ti++)
    {
        TILE_COL:
        for(int tj = 0; tj < N_TILE_COLS; tj++)
        {
            KERNEL_GRP:
            for (int tk = 0; tk < KERNEL_GROUPS; tk++)
            {
                    // Load layer bias
                BIAS: for (int f = 0; f < OUT_BUF_DEPTH; f++)
                        bias_buf[f] = layer_bias[tk*OUT_BUF_DEPTH + f];

                TILE_LYR:
                for (int tl = 0; tl < N_TILE_LAYERS; tl++)
                {
                    // Load input tile
                INP_D: for (int c = 0; c < IN_BUF_DEPTH; c++)
                        INP_R: for (int i = 0; i < BUF_HEIGHT; i++)
                            INP_C: for (int j = 0; j < BUF_WIDTH; j++)
                            {
                                #pragma HLS pipeline II=1
                                int idx_d = tl*IN_BUF_DEPTH + c;
                                int idx_h = 2*ti*BUF_HEIGHT + 2*i;
                                int idx_w = 2*tj*BUF_WIDTH + 2*j;
                                int idx = conv_ds::index_calc(idx_d, idx_h, idx_w, IN_FM_HEIGHT, IN_FM_WIDTH);
                                in_buf[c][i][j] = in_feature_map[idx];
                            }

                    // Load layer weights
                KER_OUT: for (int f = 0; f < OUT_BUF_DEPTH; f++)
                        KER_IN: for (int c = 0; c < IN_BUF_DEPTH; c++)
                        {
                            #pragma HLS pipeline II=1
                            wt_buf[f][c] = layer_weights[(tk*OUT_BUF_DEPTH + f)*IN_FM_DEPTH + tl*IN_BUF_DEPTH + c];
                        }

                    // Compute
                     conv_ds::conv_small(out_buf, in_buf, wt_buf, bias_buf, tl);
                }

                // Store output tile
            OUT_D: for (int f = 0; f < OUT_BUF_DEPTH; f++)
                    OUT_R: for (int i = 0; i < BUF_HEIGHT; i++)
                        OUT_C: for (int j = 0; j < BUF_WIDTH; j++)
                        {
                            #pragma HLS pipeline II=1
                            int idx_d = tk*OUT_BUF_DEPTH + f;
                            int idx_h = ti*BUF_HEIGHT + i;
                            int idx_w = tj*BUF_WIDTH + j;
                            int idx = conv_ds::index_calc(idx_d, idx_h, idx_w, OUT_FM_HEIGHT, OUT_FM_WIDTH);
                            out_feature_map[idx] = out_buf[f][i][j];
                        }
            }
            
        }
    }
}


// Test synthesis function
void test_conv(
    fm_t out_feature_map[],
    fm_t in_feature_map[],
    wt_t layer_weights[],
    wt_t layer_bias[]
)
{
    conv_ds::tiled_conv_ds<128, 64, 56, 56>((fm_t*) out_feature_map, (fm_t*) in_feature_map, (wt_t*) layer_weights, (wt_t*) layer_bias);
    conv_ds::tiled_conv_ds<256, 128, 28, 28>((fm_t*) out_feature_map, (fm_t*) in_feature_map, (wt_t*) layer_weights, (wt_t*) layer_bias);
    conv_ds::tiled_conv_ds<512, 256, 14, 14>((fm_t*) out_feature_map, (fm_t*) in_feature_map, (wt_t*) layer_weights, (wt_t*) layer_bias);
}

}
