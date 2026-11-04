#ifndef __MESSAGE_PASSING_H__
#define __MESSAGE_PASSING_H__

#include "dcl.h"
#include "hls_stream.h"

void message_passing_pe(
    int pe_id,
    hls::stream<mp_in_t> embeddings_per_node[NODE_PARALLEL],
    FM_TYPE message[ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM],
    int num_of_nodes
);

#endif
