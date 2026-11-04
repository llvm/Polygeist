#ifndef __DCL_H__
#define __DCL_H__

#include "util.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ap_fixed.h>
#include <array>

// #region Model Parameters
constexpr int MAX_EDGE = 90000;
constexpr int MAX_NODE = 20000;
constexpr int ND_FEATURE = 9;
constexpr int EDGE_ATTR = 3;
constexpr int EMB_DIM = 100;
constexpr int NUM_TASK = 1;
constexpr int L_IN = 200;
constexpr int L_OUT = 100;
constexpr int NUM_LAYERS = 4;
// #endregion

// #region Hardware Parameters
constexpr int LOAD_IN_EMB_PARALLEL = 2;
constexpr int SCATTER_PARALLEL = 8; // how many dimensions of EMB_DIM should a message passing PE process each cycle?
constexpr int APPLY_PARALLEL = 2; // how many dimensions of EMB_DIM should the node embedding PE process each cycle?
constexpr int NODE_PARALLEL = 2; // how many nodes should the node embedding PE process simultaneously?
constexpr int EDGE_PARALLEL = 4; // how many message passing PEs are there?
constexpr int MLP_PARALLEL = 2;
// #endregion

// #region Analysis Parameters
// actual min/avg/max from MolHIV dataset
// constexpr int ANALYSIS_NUM_GRAPHS = 4113;
// constexpr int ANALYSIS_MIN_NODES = 6;
// constexpr int ANALYSIS_AVG_NODES = 25;
// constexpr int ANALYSIS_MAX_NODES = 183;
// constexpr int ANALYSIS_MIN_EDGES = 12;
// constexpr int ANALYSIS_AVG_EDGES = 56;
// constexpr int ANALYSIS_MAX_EDGES = 378;

// g1
constexpr int ANALYSIS_NUM_GRAPHS = 1;
constexpr int ANALYSIS_MIN_NODES = 19;
constexpr int ANALYSIS_AVG_NODES = 19;
constexpr int ANALYSIS_MAX_NODES = 19;
constexpr int ANALYSIS_MIN_EDGES = 40;
constexpr int ANALYSIS_AVG_EDGES = 40;
constexpr int ANALYSIS_MAX_EDGES = 40;
// #endregion

// #region Data Types
typedef ap_fixed<16, 3> FM_TYPE;
typedef ap_fixed<16, 3> WT_TYPE;

typedef std::array<FM_TYPE, APPLY_PARALLEL> ne_out_t;
typedef std::array<FM_TYPE, SCATTER_PARALLEL> mp_in_t;
typedef std::array<FM_TYPE, MLP_PARALLEL> mlp_xfer_t;

typedef struct {
    int u;
    int v;
} edge_t;

typedef std::array<int, ND_FEATURE> node_feature_t;
typedef std::array<WT_TYPE, 4> node_eigen_t;
// #endregion

// #region Function Declarations
extern "C" {
void DGN_compute_graphs(
    int num_graphs,
    int* nums_of_nodes,
    int* nums_of_edges,
    int* reload_weights,
    FM_TYPE* out,
    node_feature_t* node_feature_in,
    node_eigen_t* node_eigen_in,
    edge_t* edge_list_in,
    WT_TYPE embedding_h_atom_embedding_list_weights_in[][9][119][100],
    WT_TYPE layers_posttrans_fully_connected_0_linear_weight_in[][4][100][200],
    WT_TYPE layers_posttrans_fully_connected_0_linear_bias_in[][4][100],
    WT_TYPE MLP_layer_FC_layers_0_weight_in[][50][100],
    WT_TYPE MLP_layer_FC_layers_0_bias_in[][50],
    WT_TYPE MLP_layer_FC_layers_1_weight_in[][25][50],
    WT_TYPE MLP_layer_FC_layers_1_bias_in[][25],
    WT_TYPE MLP_layer_FC_layers_2_weight_in[][1][25],
    WT_TYPE MLP_layer_FC_layers_2_bias_in[][1]
);
}
// #endregion

// #region Global Variables
extern WT_TYPE layers_posttrans_fully_connected_0_linear_weight[4][100][2][100];
extern WT_TYPE layers_posttrans_fully_connected_0_linear_bias[4][100];
extern WT_TYPE MLP_layer_FC_layers_0_weight[50][100];
extern WT_TYPE MLP_layer_FC_layers_0_bias[50];
extern WT_TYPE MLP_layer_FC_layers_1_weight[25][50];
extern WT_TYPE MLP_layer_FC_layers_1_bias[25];
extern WT_TYPE MLP_layer_FC_layers_2_weight[1][25];
extern WT_TYPE MLP_layer_FC_layers_2_bias[1];

extern int degree_table[MAX_NODE];
extern int degree_tables[EDGE_PARALLEL][MAX_NODE][2];
extern int neighbor_tables[EDGE_PARALLEL][MAX_EDGE];
extern WT_TYPE eig_w[EDGE_PARALLEL][MAX_EDGE];
extern WT_TYPE eig_abssums[MAX_NODE];
extern WT_TYPE eigw_sums[MAX_NODE];
extern int num_of_edges_per_pe[EDGE_PARALLEL];

// BRAM for intermediate storage
extern FM_TYPE messages_ping[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM];
extern FM_TYPE messages_pong[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM];

extern FM_TYPE h_node[MAX_NODE][EMB_DIM];
// #endregion

#endif
