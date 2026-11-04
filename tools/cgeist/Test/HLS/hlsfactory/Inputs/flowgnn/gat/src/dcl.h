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
#include <hls_vector.h>

// #region Model Parameters
constexpr int MAX_EDGE = 90000;
constexpr int MAX_NODE = 20000;
constexpr int ND_FEATURE = 9;
constexpr int ND_FEATURE_TOTAL = 173;
constexpr int EDGE_ATTR = 3;
constexpr int ED_FEATURE_PER_LAYER = 13;
constexpr int EMB_DIM = 16;
constexpr int NUM_HEADS = 4;
constexpr int NUM_LAYERS = 5;
constexpr int NUM_TASK = 1;
// #endregion

// #region Hardware Parameters
constexpr int GATHER_PARALLEL = 8; // how many dimensions of EMB_DIM should a message passing PE process each cycle?
constexpr int APPLY_PARALLEL = 1; // how many dimensions of EMB_DIM should the node embedding PE process each cycle?
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

typedef hls::vector<FM_TYPE, NUM_HEADS> FM_VEC;
typedef hls::vector<WT_TYPE, NUM_HEADS> WT_VEC;

typedef hls::vector<FM_VEC, GATHER_PARALLEL> mp_out_t;
typedef hls::vector<FM_VEC, APPLY_PARALLEL> ne_in_t;
typedef hls::vector<FM_TYPE, MLP_PARALLEL> mlp_xfer_t;

typedef hls::vector<int, ND_FEATURE> node_feature_t;
typedef hls::vector<int, EDGE_ATTR> edge_attr_t;

typedef struct {
    int u;
    int v;
} edge_t;
// #endregion

// #region Function Declarations
extern "C" {
void GAT_compute_graphs(
    int num_graphs,
    int* nums_of_nodes,
    int* nums_of_edges,
    int* reload_weights,
    FM_TYPE out[][NUM_TASK],
    node_feature_t* node_feature_in,
    edge_t* edge_list_in,
    WT_TYPE scoring_fn_target_in[][NUM_LAYERS][NUM_HEADS][EMB_DIM],
    WT_TYPE scoring_fn_source_in[][NUM_LAYERS][NUM_HEADS][EMB_DIM],
    WT_TYPE linear_proj_weights_in[][NUM_LAYERS][NUM_HEADS][EMB_DIM][NUM_HEADS][EMB_DIM],
    WT_TYPE skip_proj_weights_in[][NUM_LAYERS][NUM_HEADS][EMB_DIM][NUM_HEADS][EMB_DIM],
    WT_TYPE graph_pred_weights_in[][NUM_TASK][EMB_DIM],
    WT_TYPE graph_pred_bias_in[][NUM_TASK]
);
}
// #endregion

// #region Global Variables
extern WT_TYPE scoring_fn_target[NUM_LAYERS][NUM_HEADS][EMB_DIM];
extern WT_TYPE scoring_fn_source[NUM_LAYERS][NUM_HEADS][EMB_DIM];
extern WT_TYPE linear_proj_weights[NUM_LAYERS][NUM_HEADS][EMB_DIM][NUM_HEADS][EMB_DIM];
extern WT_TYPE skip_proj_weights[NUM_LAYERS][NUM_HEADS][EMB_DIM][NUM_HEADS][EMB_DIM];
extern WT_TYPE graph_pred_weights[NUM_TASK][EMB_DIM];
extern WT_TYPE graph_pred_bias[NUM_TASK];

extern int degree_table[MAX_NODE];
extern int degree_tables[EDGE_PARALLEL][MAX_NODE];
extern int neighbor_tables[EDGE_PARALLEL][MAX_EDGE];
extern int num_of_edges_per_pe[EDGE_PARALLEL];

// BRAM for intermediate storage
extern FM_VEC h_node_ping[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM];
extern FM_VEC h_node_pong[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM];
extern FM_VEC out_nodes_features_skip_concat_bias_ping[MAX_NODE][EMB_DIM];
extern FM_VEC out_nodes_features_skip_concat_bias_pong[MAX_NODE][EMB_DIM];
extern FM_VEC scores_source_ping[EDGE_PARALLEL][MAX_NODE];
extern FM_VEC scores_source_pong[EDGE_PARALLEL][MAX_NODE];
extern FM_VEC scores_target_ping[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)];
extern FM_VEC scores_target_pong[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)];
// #endregion

#endif
