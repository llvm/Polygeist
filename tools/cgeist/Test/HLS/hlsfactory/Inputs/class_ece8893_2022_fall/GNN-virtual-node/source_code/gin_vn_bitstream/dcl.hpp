#ifndef __DCL_H__
#define __DCL_H__

// TODO: Check why this is required
//#include "/tools/reconfig/xilinx/Vitis_HLS/2020.2/include/gmp.h"
#include "/tools/software/xilinx/Vitis_HLS/2021.1/include/gmp.h"

// Include required headers
#include <cstddef>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ap_fixed.h>
#include <iostream>
#include <fstream>

// Not relevant
// #include <math.h>
// #include <cmath>
// #include <vector>
// #include <algorithm>
// #include "hls_stream.h"

// Fixed point type suitable for HLS
typedef ap_fixed<32, 10> FM_TYPE;
typedef ap_fixed<32, 10> WT_TYPE;

/////////// Model Specific Configurations /////////////
#define MAX_EDGE 40
#define MAX_NODE 19

// For neighbor table -> 
// Max bound on the number of neighbors in any graph
#define MAX_DEGREE 4

#define LAYER_NUM 1
#define VN_LAYER_NUM 1
#define EMB_DIM 5
#define NUM_TASK 1
#define MLP_1_IN 5
#define MLP_1_OUT 10
#define MLP_2_IN 10
#define MLP_2_OUT 5

// Virtual Node Macros
#define VN_MLP_1_IN 5
#define VN_MLP_1_OUT 10
#define VN_MLP_2_IN 10
#define VN_MLP_2_OUT 5

// Unused variables (used in GoldenC)
// #define MLP_IN_MAX 200
// #define MLP_OUT_MAX 200

// Specific to GIN 
#define E_EPS 0.00001

#define ND_FEATURE 9
#define EDGE_ATTR 3

extern int nd_feature_table[ND_FEATURE]; // = {119, 4, 12, 12, 10, 6, 6, 2, 2};
#define ND_FEATURE_TOTAL 173 // 119 + 4 + ... + 2
extern int ed_feature_table[EDGE_ATTR]; // = {5, 6, 2};

#define EG_FEATURE_TOTAL 13 // (5 + 6 + 2) * LAYER_NUM

/////////// Model weights /////////////

extern float gnn_node_mlp_1_weights[LAYER_NUM][MLP_1_OUT][MLP_1_IN];
extern float gnn_node_mlp_1_bias[LAYER_NUM][MLP_1_OUT];
extern float gnn_node_mlp_2_weights[LAYER_NUM][MLP_2_OUT][MLP_2_IN];
extern float gnn_node_mlp_2_bias[LAYER_NUM][MLP_2_OUT];
extern float gnn_node_embedding_table[ND_FEATURE_TOTAL][EMB_DIM];
extern float gnn_edge_embedding_table[EG_FEATURE_TOTAL][EMB_DIM];
extern float graph_pred_linear_weight[NUM_TASK][MLP_2_OUT];
extern float graph_pred_linear_bias[NUM_TASK];
extern float eps[LAYER_NUM];

// Virtual node weights
extern float gnn_node_virtualnode_embedding_weight[1][EMB_DIM];
extern float gnn_node_virtualnode_mlp_1_weights[VN_LAYER_NUM][VN_MLP_1_OUT][VN_MLP_1_IN];
extern float gnn_node_virtualnode_mlp_1_bias[VN_LAYER_NUM][VN_MLP_1_OUT];
extern float gnn_node_virtualnode_mlp_2_weights[VN_LAYER_NUM][VN_MLP_2_OUT][VN_MLP_2_IN];
extern float gnn_node_virtualnode_mlp_2_bias[VN_LAYER_NUM][VN_MLP_2_OUT];

void load_weights();
void fetch_one_graph(char* graph_name, int* node_feature, int* edge_list, int* edge_attr, int num_of_nodes, int num_of_edges);

extern "C" {
void GIN_virtualnode_compute_one_graph(
    int* node_feature_in, int* edge_list_in, int* edge_attr_in, int* graph_attr, FM_TYPE* task,
    WT_TYPE gnn_node_mlp_1_weights_fixed[LAYER_NUM * MLP_1_OUT * MLP_1_IN],
    WT_TYPE gnn_node_mlp_1_bias_fixed[LAYER_NUM * MLP_1_OUT],
    WT_TYPE gnn_node_mlp_2_weights_fixed[LAYER_NUM * MLP_2_OUT * MLP_2_IN],
    WT_TYPE gnn_node_mlp_2_bias_fixed[LAYER_NUM * MLP_2_OUT],
    WT_TYPE gnn_node_embedding_fixed[ND_FEATURE_TOTAL * EMB_DIM],
    WT_TYPE gnn_edge_embedding_fixed[EG_FEATURE_TOTAL * EMB_DIM],
    WT_TYPE graph_pred_linear_weight_fixed[NUM_TASK * MLP_2_OUT],
    WT_TYPE graph_pred_linear_bias_fixed[NUM_TASK],
    WT_TYPE eps_fixed[LAYER_NUM],
    FM_TYPE gnn_node_virtualnode_embedding_weight_fixed[1 * EMB_DIM],
    WT_TYPE gnn_node_virtualnode_mlp_1_weights_fixed[VN_LAYER_NUM * VN_MLP_1_OUT * VN_MLP_1_IN],
    WT_TYPE gnn_node_virtualnode_mlp_1_bias_fixed[VN_LAYER_NUM * VN_MLP_1_OUT],
    WT_TYPE gnn_node_virtualnode_mlp_2_weights_fixed[VN_LAYER_NUM * VN_MLP_2_OUT * VN_MLP_2_IN], 
    WT_TYPE gnn_node_virtualnode_mlp_2_bias_fixed[VN_LAYER_NUM * VN_MLP_2_OUT]
    );
}

#endif
