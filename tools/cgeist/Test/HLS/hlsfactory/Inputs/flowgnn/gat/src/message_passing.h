#ifndef __MESSAGE_PASSING_H__
#define __MESSAGE_PASSING_H__

#include "dcl.h"
#include "hls_stream.h"

void message_passing_pe(
    int pe_id,
    FM_VEC h_node[ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_VEC scores_source[MAX_NODE],
    FM_VEC scores_target[ceildiv(MAX_NODE, EDGE_PARALLEL)],
    hls::stream<mp_out_t> messages[NODE_PARALLEL],
    hls::stream<FM_VEC> score_sums[NODE_PARALLEL],
    int num_of_nodes
);

#endif
