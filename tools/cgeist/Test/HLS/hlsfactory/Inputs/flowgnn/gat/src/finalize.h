#ifndef __FINALIZE_H__
#define __FINALIZE_H__

#include "dcl.h"
#include "hls_stream.h"

void finalize(
    hls::stream<ne_in_t> messages[NODE_PARALLEL],
    FM_VEC out_nodes_features_skip_concat_bias[MAX_NODE][EMB_DIM],
    WT_TYPE graph_pred_weights[NUM_TASK][EMB_DIM],
    WT_TYPE graph_pred_bias[NUM_TASK],
    FM_TYPE* result,
    int num_of_nodes
);

#endif
