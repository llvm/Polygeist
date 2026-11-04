#ifndef __LOAD_INPUTS_H__
#define __LOAD_INPUTS_H__

#include "dcl.h"
#include "hls_stream.h"

void load_weights(
    WT_TYPE node_mlp_1_weights_in[NUM_LAYERS][MLP_1_OUT][EMB_DIM],
    WT_TYPE node_mlp_1_bias_in[NUM_LAYERS][MLP_1_OUT],
    WT_TYPE node_mlp_2_weights_in[NUM_LAYERS][EMB_DIM][MLP_1_OUT],
    WT_TYPE node_mlp_2_bias_in[NUM_LAYERS][EMB_DIM],
    WT_TYPE edge_embedding_weight_in[NUM_LAYERS][ED_FEATURE_PER_LAYER][EMB_DIM],
    WT_TYPE graph_pred_weights_in[NUM_TASK][EMB_DIM],
    WT_TYPE graph_pred_bias_in[NUM_TASK]
);

void load_graph(
    edge_t* edge_list_in,
    edge_attr_t* edge_attr_in,
    int num_of_nodes,
    int num_of_edges
);

void load_input_node_embeddings(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    node_feature_t* node_feature,
    WT_TYPE node_embedding_weight[ND_FEATURE_TOTAL][EMB_DIM],
    FM_TYPE messages[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    int num_of_nodes
);

#endif
