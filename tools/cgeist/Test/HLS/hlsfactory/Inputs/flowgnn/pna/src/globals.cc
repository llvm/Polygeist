#include "dcl.h"

std::array<std::array<WT_TYPE, NUM_AGGRS>, NUM_SCALERS> node_conv_weights[NUM_LAYERS][EMB_DIM][EMB_DIM];
WT_TYPE node_conv_bias[NUM_LAYERS][EMB_DIM];
WT_TYPE graph_mlp_1_weights[GRAPH_MLP_1_OUT][EMB_DIM];
WT_TYPE graph_mlp_1_bias[GRAPH_MLP_1_OUT];
WT_TYPE graph_mlp_2_weights[GRAPH_MLP_2_OUT][GRAPH_MLP_1_OUT];
WT_TYPE graph_mlp_2_bias[GRAPH_MLP_2_OUT];
WT_TYPE graph_mlp_3_weights[NUM_TASK][GRAPH_MLP_2_OUT];
WT_TYPE graph_mlp_3_bias[NUM_TASK];
WT_TYPE avg_deg;

int in_degree_table[MAX_NODE];
int out_degree_tables[EDGE_PARALLEL][MAX_NODE];
int neighbor_tables[EDGE_PARALLEL][MAX_EDGE];
int num_of_edges_per_pe[EDGE_PARALLEL];
FM_TYPE log_degrees[MAX_NODE];

// BRAM for intermediate storage
std::array<FM_TYPE, NUM_AGGRS> messages_ping[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM];
std::array<FM_TYPE, NUM_AGGRS> messages_pong[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM];

FM_TYPE h_node[MAX_NODE][EMB_DIM];
