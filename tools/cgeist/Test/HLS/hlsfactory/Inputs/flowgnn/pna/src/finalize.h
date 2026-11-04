#ifndef __FINALIZE_H__
#define __FINALIZE_H__

#include "dcl.h"
#include "hls_stream.h"

void finalize(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    WT_TYPE graph_mlp_1_weights[GRAPH_MLP_1_OUT][EMB_DIM],
    WT_TYPE graph_mlp_1_bias[GRAPH_MLP_1_OUT],
    WT_TYPE graph_mlp_2_weights[GRAPH_MLP_2_OUT][GRAPH_MLP_1_OUT],
    WT_TYPE graph_mlp_2_bias[GRAPH_MLP_2_OUT],
    WT_TYPE graph_mlp_3_weights[NUM_TASK][GRAPH_MLP_2_OUT],
    WT_TYPE graph_mlp_3_bias[NUM_TASK],
    FM_TYPE* result,
    int num_of_nodes
);

#endif
