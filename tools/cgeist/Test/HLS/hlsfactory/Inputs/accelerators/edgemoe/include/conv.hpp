#ifndef __CONV_HPP__
#define __CONV_HPP__

#include "dcl.hpp"

extern wt_patch_embed_t patch_embed_weights[FEATURE_DIM][INPUT_CHANNELS][PATCH_HEIGHT][PATCH_WIDTH];
extern wt_bias_t patch_embed_bias[FEATURE_DIM];

void compute_patch_embed(image_t x, patch_blocks_t out, patch_blocks_t pos_embed);

#endif
