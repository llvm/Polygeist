#ifndef __TESTBENCH_H__
#define __TESTBENCH_H__

#include "../src/dcl.h"

constexpr int NUM_GRAPHS = 1;

extern WT_TYPE (*embedding_h_atom_embedding_list_weights)[9][119][100];
extern WT_TYPE (*layers_posttrans_fully_connected_0_linear_weight_in)[4][100][200];
extern WT_TYPE (*layers_posttrans_fully_connected_0_linear_bias_in)[4][100];
extern WT_TYPE (*MLP_layer_FC_layers_0_weight_in)[50][100];
extern WT_TYPE (*MLP_layer_FC_layers_0_bias_in)[50];
extern WT_TYPE (*MLP_layer_FC_layers_1_weight_in)[25][50];
extern WT_TYPE (*MLP_layer_FC_layers_1_bias_in)[25];
extern WT_TYPE (*MLP_layer_FC_layers_2_weight_in)[1][25];
extern WT_TYPE (*MLP_layer_FC_layers_2_bias_in)[1];

void load_weights();
void fetch_one_graph(
    int g,
    char* graph_name,
    node_feature_t* node_feature,
    node_eigen_t* node_eigen,
    edge_t* edge_list,
    int* edge_attr,
    int num_of_nodes,
    int num_of_edges
);

#endif
