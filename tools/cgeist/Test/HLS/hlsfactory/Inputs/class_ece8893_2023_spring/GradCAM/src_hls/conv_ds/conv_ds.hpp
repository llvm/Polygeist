#pragma once

#include "../util.h"

namespace conv_ds
{

const int IN_BUF_DEPTH = 16;
const int OUT_BUF_DEPTH = 16;
const int BUF_HEIGHT = 7;
const int BUF_WIDTH = 7;

void tiled_conv_ds_core(
    fm_t out_feature_map[],
    const fm_t in_feature_map[],
    const wt_t layer_weights[],
    const wt_t layer_bias[],
    const int N_TILE_ROWS,
    const int N_TILE_COLS,
    const int N_TILE_LAYERS,
    const int KERNEL_GROUPS
);

template<int OUT_FM_DEPTH, int IN_FM_DEPTH, int IN_FM_HEIGHT, int IN_FM_WIDTH>
void tiled_conv_ds(
    fm_t out_feature_map[],
    const fm_t in_feature_map[],
    const wt_t layer_weights[],
    const wt_t layer_bias[]
)
{
    #pragma HLS inline
    const int N_TILE_ROWS = IN_FM_HEIGHT / (2 * BUF_HEIGHT);
    const int N_TILE_COLS = IN_FM_WIDTH / (2 * BUF_WIDTH);
    const int N_TILE_LAYERS = IN_FM_DEPTH / IN_BUF_DEPTH;
    const int KERNEL_GROUPS = OUT_FM_DEPTH / OUT_BUF_DEPTH;

    conv_ds::tiled_conv_ds_core(
        (fm_t *) out_feature_map,
        (fm_t *) in_feature_map,
        (fm_t *) layer_weights,
        layer_bias,
        N_TILE_ROWS,
        N_TILE_COLS,
        N_TILE_LAYERS,
        KERNEL_GROUPS
    );
}

}
