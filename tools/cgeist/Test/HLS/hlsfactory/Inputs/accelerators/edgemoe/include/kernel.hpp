#ifndef __KERNEL_HPP__
#define __KERNEL_HPP__

// https://support.xilinx.com/s/question/0D52E00006iHkfp/vivado-20153-hls-bug-gmph?language=en_US
#include <gmp.h>
#define __gmp_const const

#include "dcl.hpp"
#include "attention.hpp"
#include "layernorm.hpp"

extern "C"
{
    void ViT_compute(
        unsigned int num_images,
        bool reload_one_time_weights,
        image_t images[],
        patch_blocks_t x[],
        patch_blocks_t tmp1,
        patch_blocks_t tmp2,
        patch_blocks_t tmp3,
        patch_blocks_t tmp4,
        fm_block_t tmp_hidden[NUM_PATCHES * ceildiv(max(VIT_HIDDEN_DIM, EXPERT_HIDDEN_DIM), FEATURE_BLOCK_SIZE)],
        qxk_out_t attn,
        softmax_info_t attn_softmax_info,
        wt_patch_embed_t patch_embed_weights[FEATURE_DIM][INPUT_CHANNELS][PATCH_HEIGHT][PATCH_WIDTH],
        wt_bias_t patch_embed_bias[FEATURE_DIM],
        patch_blocks_t pos_embed,
        wt_linear_t attn_weights[NUM_LAYERS][NUM_ATTN_LINEAR][FEATURE_DIM][FEATURE_DIM],
        wt_attn_bias_t attn_bias[NUM_LAYERS][NUM_ATTN_LINEAR][FEATURE_DIM],
        wt_linear_t moe_w_gate[max(NUM_LAYERS / 2, 1U)][NUM_EXPERTS][FEATURE_DIM],
        wt_linear_t moe_weights_l1[max(NUM_LAYERS / 2, 1U)][NUM_EXPERTS][EXPERT_HIDDEN_DIM][FEATURE_DIM],
        wt_bias_t moe_bias_l1[max(NUM_LAYERS / 2, 1U)][NUM_EXPERTS][EXPERT_HIDDEN_DIM],
        wt_linear_t moe_weights_l2[max(NUM_LAYERS / 2, 1U)][NUM_EXPERTS][FEATURE_DIM][EXPERT_HIDDEN_DIM],
        wt_bias_t moe_bias_l2[max(NUM_LAYERS / 2, 1U)][NUM_EXPERTS][FEATURE_DIM],
        wt_linear_t vit_weights_l1[max((NUM_LAYERS + 1) / 2, 1U)][VIT_HIDDEN_DIM][FEATURE_DIM],
        wt_bias_t vit_bias_l1[max((NUM_LAYERS + 1) / 2, 1U)][VIT_HIDDEN_DIM],
        wt_linear_t vit_weights_l2[max((NUM_LAYERS + 1) / 2, 1U)][FEATURE_DIM][VIT_HIDDEN_DIM],
        wt_bias_t vit_bias_l2[max((NUM_LAYERS + 1) / 2, 1U)][FEATURE_DIM],
        wt_norm_t norm_weights[NUM_LAYERS][NUM_LAYER_NORMS][FEATURE_DIM],
        wt_bias_t norm_bias[NUM_LAYERS][NUM_LAYER_NORMS][FEATURE_DIM],
        unsigned int debug_id
    );
}

#endif
