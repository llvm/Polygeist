#ifndef __LOAD_INPUTS_H__
#define __LOAD_INPUTS_H__

#include "dcl.h"
#include "hls_stream.h"

void load_weights(
    WT_TYPE convs_weight_in[NUM_LAYERS][100][100],
    WT_TYPE convs_bias_in[NUM_LAYERS][100],
    WT_TYPE convs_root_emb_weight_in[NUM_LAYERS][100],
    WT_TYPE bn_weight_in[NUM_LAYERS][100],
    WT_TYPE bn_bias_in[NUM_LAYERS][100],
    WT_TYPE bn_mean_in[NUM_LAYERS][100],
    WT_TYPE bn_var_in[NUM_LAYERS][100],
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
    node_feature_t* node_feature,
    WT_TYPE node_embedding_weight[ND_FEATURE_TOTAL][EMB_DIM],
    FM_TYPE messages[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    int num_of_nodes
);

#endif
