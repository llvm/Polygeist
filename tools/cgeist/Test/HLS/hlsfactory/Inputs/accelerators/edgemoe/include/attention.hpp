#ifndef __ATTENTION_HPP__
#define __ATTENTION_HPP__

#include "dcl.hpp"
#include "linear.hpp"

enum AttentionLinear {
    ATTN_Q = 0,
    ATTN_K = 1,
    ATTN_V = 2,
    ATTN_PROJ = 3,
    NUM_ATTN_LINEAR
};

extern fm_t attn_scale;

void compute_q_matmul_k(
    patch_blocks_t q,
    patch_blocks_t k,
    qxk_out_t attn,
    softmax_info_t attn_softmax_info
);
void compute_attn_matmul_v(
    patch_blocks_t v,
    qxk_out_t attn,
    softmax_info_t attn_softmax_info,
    patch_blocks_t attn_matmul_v
);

#endif
