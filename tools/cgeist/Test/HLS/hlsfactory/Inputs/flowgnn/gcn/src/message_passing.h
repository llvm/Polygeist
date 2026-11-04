#ifndef __MESSAGE_PASSING_H__
#define __MESSAGE_PASSING_H__

#include "dcl.h"
#include "hls_stream.h"

void message_passing_pe(
    int pe_id,
    hls::stream<mp_in_t> node_embeddings[NODE_PARALLEL],
    FM_TYPE message[ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    int layer_num,
    int num_of_nodes
);

#endif
