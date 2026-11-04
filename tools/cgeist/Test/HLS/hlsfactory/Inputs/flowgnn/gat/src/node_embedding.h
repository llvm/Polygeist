#ifndef __NODE_EMBEDDING_H__
#define __NODE_EMBEDDING_H__

#include "dcl.h"
#include "hls_stream.h"

void node_embedding_multi_pe(
    hls::stream<ne_in_t> messages[NODE_PARALLEL],
    FM_VEC h_node[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_VEC out_nodes_features_skip_concat_bias[MAX_NODE][EMB_DIM],
    FM_VEC next_out_nodes_features_skip_concat_bias[MAX_NODE][EMB_DIM],
    FM_VEC scores_source[EDGE_PARALLEL][MAX_NODE],
    FM_VEC scores_target[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)],
    int layer_num,
    int num_of_nodes
);

#endif
