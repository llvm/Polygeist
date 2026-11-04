#pragma once

#include "../util.h"

namespace conv_3x3_s1 {

#include "params.hpp"

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
);

template<int OUT_FM_DEPTH, int IN_FM_DEPTH, int IN_FM_HEIGHT, int IN_FM_WIDTH>
inline void tiled_conv (
    fm_t output_feature_map[],
    const fm_t input_feature_map[],
    const wt_t layer_weights[],
    const wt_t layer_bias[],
    const bool inplace_residual = false,
    const bool stride_2 = false
)
{ 
    #pragma HLS inline

    //static_assert(IN_FM_DEPTH >= IN_BUF_DEPTH, "IN_FM_WIDTH >= IN_BUF_DEPTH");

    const int KERNEL_GRPS = OUT_FM_DEPTH / OUT_BUF_DEPTH;
    const int N_TILE_ROWS = IN_FM_HEIGHT / TILE_HEIGHT;
    const int N_TILE_COLS = IN_FM_WIDTH  / TILE_WIDTH;
    int N_TILE_LAYERS = IN_FM_DEPTH / IN_BUF_DEPTH;
    if (N_TILE_LAYERS == 0) N_TILE_LAYERS = 1;

    conv_3x3_s1::tiled_conv_core(
        output_feature_map,
        input_feature_map,
        (fm_t *) layer_weights,
        (fm_t *) layer_bias,
        KERNEL_GRPS,
        IN_FM_DEPTH,
        N_TILE_LAYERS,
        N_TILE_ROWS,
        N_TILE_COLS,
        stride_2,
        inplace_residual
        );

}

}
