#ifndef __NODE_EMBEDDING_H__
#define __NODE_EMBEDDING_H__

#include "dcl.h"
#include "hls_stream.h"

void node_embedding_multi_pe(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    FM_TYPE message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    int layer_num,
    int num_of_nodes
);

#endif
