#ifndef __CONV_LAYER_H__
#define __CONV_LAYER_H__

#include "dcl.h"

void compute_CONV_layer(
    int layer_num,
    FM_TYPE message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM],
    FM_TYPE next_message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM],
    node_feature_t* node_feature_in,
    WT_TYPE embedding_h_atom_embedding_list_weights_in[9][119][100],
    FM_TYPE* result,
    int num_of_nodes
);

#endif
