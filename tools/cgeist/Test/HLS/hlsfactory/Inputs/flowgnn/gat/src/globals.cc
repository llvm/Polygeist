#include "dcl.h"

WT_TYPE scoring_fn_target[NUM_LAYERS][NUM_HEADS][EMB_DIM];
WT_TYPE scoring_fn_source[NUM_LAYERS][NUM_HEADS][EMB_DIM];
WT_TYPE linear_proj_weights[NUM_LAYERS][NUM_HEADS][EMB_DIM][NUM_HEADS][EMB_DIM];
WT_TYPE skip_proj_weights[NUM_LAYERS][NUM_HEADS][EMB_DIM][NUM_HEADS][EMB_DIM];
WT_TYPE graph_pred_weights[NUM_TASK][EMB_DIM];
WT_TYPE graph_pred_bias[NUM_TASK];

int degree_table[MAX_NODE];
int degree_tables[EDGE_PARALLEL][MAX_NODE];
int neighbor_tables[EDGE_PARALLEL][MAX_EDGE];
int num_of_edges_per_pe[EDGE_PARALLEL];

// BRAM for intermediate storage
FM_VEC h_node_ping[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM];
FM_VEC h_node_pong[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM];
FM_VEC out_nodes_features_skip_concat_bias_ping[MAX_NODE][EMB_DIM];
FM_VEC out_nodes_features_skip_concat_bias_pong[MAX_NODE][EMB_DIM];
FM_VEC scores_source_ping[EDGE_PARALLEL][MAX_NODE];
FM_VEC scores_source_pong[EDGE_PARALLEL][MAX_NODE];
FM_VEC scores_target_ping[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)];
FM_VEC scores_target_pong[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)];
