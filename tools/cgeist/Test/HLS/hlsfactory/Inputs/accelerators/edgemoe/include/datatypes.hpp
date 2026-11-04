#ifndef __DATATYPES_HPP__
#define __DATATYPES_HPP__

#include "model.hpp"
#include <ap_fixed.h>
#include <hls_vector.h>

typedef ap_fixed<32, 10> fm_t;
typedef ap_fixed<16, 2> wt_linear_t;
typedef ap_fixed<16, 7> wt_attn_bias_t;
typedef ap_fixed<16, 5> wt_bias_t;
typedef ap_fixed<16, 5> wt_norm_t;
typedef ap_fixed<16, 0> wt_patch_embed_t;

typedef ap_ufixed<8, 0> pixel_t;
typedef pixel_t image_t[INPUT_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH];

#endif
