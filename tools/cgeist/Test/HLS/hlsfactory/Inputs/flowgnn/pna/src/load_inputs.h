#ifndef __LOAD_INPUTS_H__
#define __LOAD_INPUTS_H__

#include "dcl.h"
#include "hls_stream.h"
#include "hls_math.h"

void load_weights(
    WT_TYPE node_conv_weights_in[NUM_LAYERS][EMB_DIM][NUM_SCALERS][NUM_AGGRS][EMB_DIM],
    WT_TYPE node_conv_bias_in[NUM_LAYERS][EMB_DIM],
    WT_TYPE graph_mlp_1_weights_in[GRAPH_MLP_1_OUT][EMB_DIM],
    WT_TYPE graph_mlp_1_bias_in[GRAPH_MLP_1_OUT],
    WT_TYPE graph_mlp_2_weights_in[GRAPH_MLP_2_OUT][GRAPH_MLP_1_OUT],
    WT_TYPE graph_mlp_2_bias_in[GRAPH_MLP_2_OUT],
    WT_TYPE graph_mlp_3_weights_in[NUM_TASK][GRAPH_MLP_2_OUT],
    WT_TYPE graph_mlp_3_bias_in[NUM_TASK],
    WT_TYPE avg_deg_in
);

void load_graph(
    edge_t* edge_list_in,
    int num_of_nodes,
    int num_of_edges
);

void load_input_node_embeddings(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    node_feature_t* node_feature,
    WT_TYPE node_embedding_weight[ND_FEATURE_TOTAL][EMB_DIM],
    std::array<FM_TYPE, NUM_AGGRS> messages[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    int num_of_nodes
);

void reset_messages(
    std::array<FM_TYPE, NUM_AGGRS> messages[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    int num_of_nodes
);

#endif
