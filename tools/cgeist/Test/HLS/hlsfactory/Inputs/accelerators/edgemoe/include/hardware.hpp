#ifndef __HARDWARE_HPP__
#define __HARDWARE_HPP__

#include "datatypes.hpp"
#include "model.hpp"
#include "util.hpp"
#include "hls_vector.h"

constexpr unsigned int AXI_XFER_BIT_WIDTH = 256;
constexpr unsigned int FEATURE_BLOCK_SIZE = (AXI_XFER_BIT_WIDTH / fm_t::width);
constexpr unsigned int NUM_FEATURE_BLOCKS = ceildiv(FEATURE_DIM, FEATURE_BLOCK_SIZE);

constexpr unsigned int LINEAR_IN_SIZE = 2 * FEATURE_BLOCK_SIZE;
constexpr unsigned int LINEAR_OUT_SIZE = 2 * FEATURE_BLOCK_SIZE;
constexpr unsigned int ATTN_MATMUL_PARALLEL = 4;

typedef hls::vector<fm_t, FEATURE_BLOCK_SIZE> fm_block_t;
typedef fm_block_t fm_blocks_t[NUM_FEATURE_BLOCKS];
typedef fm_blocks_t patch_blocks_t[NUM_PATCHES];

typedef hls::vector<fm_t, LINEAR_IN_SIZE> linear_in_t;
typedef hls::vector<fm_t, LINEAR_OUT_SIZE> linear_out_t;

typedef hls::vector<fm_t, roundup_p2(NUM_HEADS)> heads_t;
typedef hls::vector<heads_t, ATTN_MATMUL_PARALLEL> attn_parallel_t;
typedef heads_t qxk_out_t[ceildiv(NUM_PATCHES + ATTN_MATMUL_PARALLEL - 1, ATTN_MATMUL_PARALLEL)][NUM_PATCHES][ATTN_MATMUL_PARALLEL];
typedef hls::vector<fm_t, roundup_p2(NUM_HEADS * 2)> softmax_info_row_t;
typedef softmax_info_row_t softmax_info_t[NUM_PATCHES];

#endif
