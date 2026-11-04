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
constexpr int MAX_EDGE = 90000;
constexpr int MAX_NODE = 20000;
constexpr int ND_FEATURE = 9;
constexpr int ND_FEATURE_TOTAL = 173;
constexpr int EDGE_ATTR = 3;
constexpr int ED_FEATURE_PER_LAYER = 13;
constexpr int EMB_DIM = 80;
constexpr int NUM_LAYERS = 4;
constexpr int NUM_TASK = 1;
constexpr int GRAPH_MLP_1_OUT = 40;
constexpr int GRAPH_MLP_2_OUT = 20;

typedef enum {
    AGGR_MEAN,
    AGGR_MIN,
    AGGR_MAX,
    AGGR_STD,
    NUM_AGGRS
} aggregator_t;

typedef enum {
    SCALER_NONE,
    SCALER_T,
    SCALER_SCALE,
    NUM_SCALERS
} scaler_t;
// #endregion

// #region Hardware Parameters
constexpr int SCATTER_PARALLEL = 8; // how many dimensions of EMB_DIM should a message passing PE process each cycle?
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

typedef std::array<FM_TYPE, APPLY_PARALLEL> ne_out_t;
typedef std::array<FM_TYPE, SCATTER_PARALLEL> mp_in_t;
typedef std::array<FM_TYPE, MLP_PARALLEL> mlp_xfer_t;

typedef std::array<int, ND_FEATURE> node_feature_t;
typedef std::array<int, EDGE_ATTR> edge_attr_t;

typedef struct {
    int u;
    int v;
} edge_t;
// #endregion

// #region Function Declarations
extern "C" {
void PNA_compute_graphs(
    int num_graphs,
    int* nums_of_nodes,
    int* nums_of_edges,
    int* reload_weights,
    FM_TYPE out[][NUM_TASK],
    node_feature_t* node_feature_in,
    edge_t* edge_list_in,
    WT_TYPE node_embedding_weight_in[][ND_FEATURE_TOTAL][EMB_DIM],
    WT_TYPE node_conv_weights_in[][NUM_LAYERS][EMB_DIM][NUM_SCALERS][NUM_AGGRS][EMB_DIM],
    WT_TYPE node_conv_bias_in[][NUM_LAYERS][EMB_DIM],
    WT_TYPE graph_mlp_1_weights_in[][GRAPH_MLP_1_OUT][EMB_DIM],
    WT_TYPE graph_mlp_1_bias_in[][GRAPH_MLP_1_OUT],
    WT_TYPE graph_mlp_2_weights_in[][GRAPH_MLP_2_OUT][GRAPH_MLP_1_OUT],
    WT_TYPE graph_mlp_2_bias_in[][GRAPH_MLP_2_OUT],
    WT_TYPE graph_mlp_3_weights_in[][NUM_TASK][GRAPH_MLP_2_OUT],
    WT_TYPE graph_mlp_3_bias_in[][NUM_TASK],
    WT_TYPE avg_deg_in[]
);
}
// #endregion

// #region Global Variables
extern std::array<std::array<WT_TYPE, NUM_AGGRS>, NUM_SCALERS> node_conv_weights[NUM_LAYERS][EMB_DIM][EMB_DIM];
extern WT_TYPE node_conv_bias[NUM_LAYERS][EMB_DIM];
extern WT_TYPE graph_mlp_1_weights[GRAPH_MLP_1_OUT][EMB_DIM];
extern WT_TYPE graph_mlp_1_bias[GRAPH_MLP_1_OUT];
extern WT_TYPE graph_mlp_2_weights[GRAPH_MLP_2_OUT][GRAPH_MLP_1_OUT];
extern WT_TYPE graph_mlp_2_bias[GRAPH_MLP_2_OUT];
extern WT_TYPE graph_mlp_3_weights[NUM_TASK][GRAPH_MLP_2_OUT];
extern WT_TYPE graph_mlp_3_bias[NUM_TASK];
extern WT_TYPE avg_deg;

extern int in_degree_table[MAX_NODE];
extern int out_degree_tables[EDGE_PARALLEL][MAX_NODE];
extern int neighbor_tables[EDGE_PARALLEL][MAX_EDGE];
extern int num_of_edges_per_pe[EDGE_PARALLEL];
extern FM_TYPE log_degrees[MAX_NODE];

// BRAM for intermediate storage
extern std::array<FM_TYPE, NUM_AGGRS> messages_ping[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM];
extern std::array<FM_TYPE, NUM_AGGRS> messages_pong[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM];

extern FM_TYPE h_node[MAX_NODE][EMB_DIM];
// #endregion

#endif
