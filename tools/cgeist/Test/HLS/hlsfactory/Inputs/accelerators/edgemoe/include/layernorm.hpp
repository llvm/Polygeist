#ifndef __LAYERNORM_HPP__
#define __LAYERNORM_HPP__

#include "dcl.hpp"

enum LayerNorm {
    NORM_1 = 0,
    NORM_2 = 1,
    NUM_LAYER_NORMS
};

extern fm_t norm_eps;

void load_norms(
    wt_norm_t norm_weights[NUM_LAYER_NORMS][FEATURE_DIM],
    wt_bias_t norm_bias[NUM_LAYER_NORMS][FEATURE_DIM]
);
void compute_norm1(patch_blocks_t x, patch_blocks_t out);
void compute_norm2(patch_blocks_t x, patch_blocks_t out);

#endif
