#ifndef __FINALIZE_H__
#define __FINALIZE_H__

#include "dcl.h"
#include "hls_stream.h"

void finalize(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    WT_TYPE MLP_layer_FC_layers_0_weight[50][100],
    WT_TYPE MLP_layer_FC_layers_0_bias[50],
    WT_TYPE MLP_layer_FC_layers_1_weight[25][50],
    WT_TYPE MLP_layer_FC_layers_1_bias[25],
    WT_TYPE MLP_layer_FC_layers_2_weight[1][25],
    WT_TYPE MLP_layer_FC_layers_2_bias[1],
    FM_TYPE* result,
    int num_of_nodes
);

#endif
