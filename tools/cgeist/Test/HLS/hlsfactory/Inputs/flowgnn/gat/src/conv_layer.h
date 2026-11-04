#ifndef __CONV_LAYER_H__
#define __CONV_LAYER_H__

#include "dcl.h"

void compute_CONV_layer(
    int layer_num,
    FM_VEC h_node[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_VEC next_h_node[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_VEC scores_source[EDGE_PARALLEL][MAX_NODE],
    FM_VEC next_scores_source[EDGE_PARALLEL][MAX_NODE],
    FM_VEC scores_target[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)],
    FM_VEC next_scores_target[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)],
    FM_VEC out_nodes_features_skip_concat_bias[MAX_NODE][EMB_DIM],
    FM_VEC next_out_nodes_features_skip_concat_bias[MAX_NODE][EMB_DIM],
    FM_TYPE* result,
    int num_of_nodes
);

#endif
