#ifndef __MESSAGE_PASSING_H__
#define __MESSAGE_PASSING_H__

#include "dcl.h"
#include "hls_stream.h"

void message_passing_pe(
    int pe_id,
    int layer_num,
    hls::stream<mp_in_t> embeddings_per_node[NODE_PARALLEL],
    FM_TYPE message[ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM]
);

#endif
