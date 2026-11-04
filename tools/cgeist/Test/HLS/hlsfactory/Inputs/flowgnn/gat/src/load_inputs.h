#ifndef __LOAD_INPUTS_H__
#define __LOAD_INPUTS_H__

#include "dcl.h"
#include "hls_stream.h"

void load_weights(
    WT_TYPE scoring_fn_target_in[NUM_LAYERS][NUM_HEADS][EMB_DIM],
    WT_TYPE scoring_fn_source_in[NUM_LAYERS][NUM_HEADS][EMB_DIM],
    WT_TYPE linear_proj_weights_in[NUM_LAYERS][NUM_HEADS][EMB_DIM][NUM_HEADS][EMB_DIM],
    WT_TYPE skip_proj_weights_in[NUM_LAYERS][NUM_HEADS][EMB_DIM][NUM_HEADS][EMB_DIM],
    WT_TYPE graph_pred_weights_in[NUM_TASK][EMB_DIM],
    WT_TYPE graph_pred_bias_in[NUM_TASK]
);

void load_graph(
    edge_t* edge_list_in,
    int num_of_nodes,
    int num_of_edges
);

void load_input_node_embeddings(node_feature_t* node_feature, int num_of_nodes);

#endif
