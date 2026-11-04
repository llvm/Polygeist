#ifndef __CONV_LAYER_H__
#define __CONV_LAYER_H__

#include "dcl.h"

void compute_CONV_layer(
    int layer_num,
    FM_TYPE message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_TYPE next_message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    node_feature_t* node_feature_in,
    WT_TYPE node_embedding_weight_in[ND_FEATURE_TOTAL][EMB_DIM],
    FM_TYPE* result,
    int num_of_nodes
);

#endif
