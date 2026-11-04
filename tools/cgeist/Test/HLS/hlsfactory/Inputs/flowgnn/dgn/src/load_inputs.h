#ifndef __LOAD_INPUTS_H__
#define __LOAD_INPUTS_H__

#include "dcl.h"
#include "hls_stream.h"

void load_weights(
    WT_TYPE layers_posttrans_fully_connected_0_linear_weight_in[4][100][200],
    WT_TYPE layers_posttrans_fully_connected_0_linear_bias_in[4][100],
    WT_TYPE MLP_layer_FC_layers_0_weight_in[50][100],
    WT_TYPE MLP_layer_FC_layers_0_bias_in[50],
    WT_TYPE MLP_layer_FC_layers_1_weight_in[25][50],
    WT_TYPE MLP_layer_FC_layers_1_bias_in[25],
    WT_TYPE MLP_layer_FC_layers_2_weight_in[1][25],
    WT_TYPE MLP_layer_FC_layers_2_bias_in[1]
);

void load_graph(
    edge_t *edge_list_in,
    node_eigen_t *node_eigen_in,
    int num_of_nodes,
    int num_of_edges
);

void load_input_node_embeddings(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    node_feature_t* node_feature,
    WT_TYPE embedding_h_atom_embedding_list_weights[9][119][100],
    int num_of_nodes
);

#endif
