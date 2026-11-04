#ifndef __MOE_HPP__
#define __MOE_HPP__

#include "dcl.hpp"

extern unsigned int expert_queues[NUM_EXPERTS][NUM_PATCHES];
extern fm_t expert_scores[NUM_EXPERTS][NUM_PATCHES];
extern unsigned int expert_queue_lens[NUM_EXPERTS];

void load_w_gate(wt_linear_t w_gate_load[NUM_EXPERTS][FEATURE_DIM]);
void compute_moe(
    patch_blocks_t gate_inp,
    patch_blocks_t out,
    fm_block_t tmp_hidden[],
    wt_linear_t all_weights_l1[NUM_EXPERTS][EXPERT_HIDDEN_DIM][FEATURE_DIM],
    wt_bias_t all_bias_l1[NUM_EXPERTS][EXPERT_HIDDEN_DIM],
    wt_linear_t all_weights_l2[NUM_EXPERTS][FEATURE_DIM][EXPERT_HIDDEN_DIM],
    wt_bias_t all_bias_l2[NUM_EXPERTS][FEATURE_DIM]
);

#endif
