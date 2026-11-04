#ifndef __FINALIZE_H__
#define __FINALIZE_H__

#include "dcl.h"
#include "hls_stream.h"

void finalize(
    FM_TYPE h_node[MAX_NODE][EMB_DIM],
    FM_TYPE message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    WT_TYPE graph_pred_weights[NUM_TASK][EMB_DIM],
    WT_TYPE graph_pred_bias[NUM_TASK],
    FM_TYPE* result,
    int num_of_nodes
);

#endif
