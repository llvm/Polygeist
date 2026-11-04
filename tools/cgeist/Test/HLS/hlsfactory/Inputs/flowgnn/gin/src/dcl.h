#ifndef __DCL_H__
#define __DCL_H__

// https://support.xilinx.com/s/question/0D52E00006iHkfp/vivado-20153-hls-bug-gmph?language=en_US
#define __gmp_const const

#include "util.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ap_fixed.h>
#include <array>

// #region Model Parameters
constexpr int MAX_EDGE = 500;
constexpr int MAX_NODE = 500;
constexpr int ND_FEATURE = 9;
constexpr int ND_FEATURE_TOTAL = 173;
constexpr int EDGE_ATTR = 3;
constexpr int ED_FEATURE_PER_LAYER = 13;
constexpr int EMB_DIM = 100;
constexpr int NUM_LAYERS = 5;
constexpr int NUM_TASK = 1;
constexpr int MLP_1_OUT = 200;
// #endregion

// #region Hardware Parameters
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
typedef ap_fixed<16, 6> FM_TYPE;
typedef ap_fixed<16, 6> WT_TYPE;

typedef std::array<FM_TYPE, APPLY_PARALLEL> ne_out_t;
typedef std::array<FM_TYPE, SCATTER_PARALLEL> mp_in_t;
typedef std::array<FM_TYPE, MLP_PARALLEL> mlp_xfer_t;

typedef std::array<int, ND_FEATURE> node_feature_t;
typedef std::array<int, EDGE_ATTR> edge_attr_t;

typedef struct {
    int nd;
    int degree;
} node_t;

typedef struct {
    int u;
    int v;
} edge_t;
// #endregion

// #region Function Declarations
extern "C" {
void GIN_compute_graphs(
    int num_graphs,
    int* nums_of_nodes,
    int* nums_of_edges,
    int* reload_weights,
    FM_TYPE out[][NUM_TASK],
    node_feature_t* node_feature_in,
    edge_t* edge_list_in,
    edge_attr_t* edge_attr_in,
    WT_TYPE node_embedding_weight_in[][ND_FEATURE_TOTAL][EMB_DIM],
    WT_TYPE edge_embedding_weight_in[][NUM_LAYERS][ED_FEATURE_PER_LAYER][EMB_DIM],
    WT_TYPE node_mlp_1_weights[][NUM_LAYERS][MLP_1_OUT][EMB_DIM],
    WT_TYPE node_mlp_1_bias[][NUM_LAYERS][MLP_1_OUT],
    WT_TYPE node_mlp_2_weights[][NUM_LAYERS][EMB_DIM][MLP_1_OUT],
    WT_TYPE node_mlp_2_bias[][NUM_LAYERS][EMB_DIM],
    WT_TYPE graph_pred_weights_in[][NUM_TASK][EMB_DIM],
    WT_TYPE graph_pred_bias_in[][NUM_TASK]
);
}
// #endregion

// #region Global Variables
extern WT_TYPE node_mlp_eps[NUM_LAYERS];
extern WT_TYPE node_mlp_1_weights[NUM_LAYERS][MLP_1_OUT][EMB_DIM];
extern WT_TYPE node_mlp_1_bias[NUM_LAYERS][MLP_1_OUT];
extern WT_TYPE node_mlp_2_weights[NUM_LAYERS][EMB_DIM][MLP_1_OUT];
extern WT_TYPE node_mlp_2_bias[NUM_LAYERS][EMB_DIM];

extern WT_TYPE edge_embedding_weights[EDGE_PARALLEL][NUM_LAYERS][ED_FEATURE_PER_LAYER][EMB_DIM];
extern WT_TYPE graph_pred_weights[NUM_TASK][EMB_DIM];
extern WT_TYPE graph_pred_bias[NUM_TASK];

extern int degree_table[MAX_NODE];
extern ap_int<EDGE_PARALLEL> pes_per_node[MAX_NODE];
extern node_t degree_tables[EDGE_PARALLEL][MAX_NODE];
extern int neighbor_tables[EDGE_PARALLEL][MAX_EDGE];
extern edge_attr_t edge_attrs[EDGE_PARALLEL][MAX_EDGE];
extern int num_of_edges_per_pe[EDGE_PARALLEL];

// BRAM for intermediate storage
extern FM_TYPE messages_ping[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM];
extern FM_TYPE messages_pong[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM];

extern FM_TYPE h_node[MAX_NODE][EMB_DIM];
// #endregion

#endif
