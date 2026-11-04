#include "dcl.h"

WT_TYPE node_mlp_eps[NUM_LAYERS];
WT_TYPE node_mlp_1_weights[NUM_LAYERS][MLP_1_OUT][EMB_DIM];
WT_TYPE node_mlp_1_bias[NUM_LAYERS][MLP_1_OUT];
WT_TYPE node_mlp_2_weights[NUM_LAYERS][EMB_DIM][MLP_1_OUT];
WT_TYPE node_mlp_2_bias[NUM_LAYERS][EMB_DIM];

WT_TYPE edge_embedding_weights[EDGE_PARALLEL][NUM_LAYERS][ED_FEATURE_PER_LAYER][EMB_DIM];
WT_TYPE graph_pred_weights[NUM_TASK][EMB_DIM];
WT_TYPE graph_pred_bias[NUM_TASK];

int degree_table[MAX_NODE];
ap_int<EDGE_PARALLEL> pes_per_node[MAX_NODE];
node_t degree_tables[EDGE_PARALLEL][MAX_NODE];
int neighbor_tables[EDGE_PARALLEL][MAX_EDGE];
edge_attr_t edge_attrs[EDGE_PARALLEL][MAX_EDGE];
int num_of_edges_per_pe[EDGE_PARALLEL];

// BRAM for intermediate storage
FM_TYPE messages_ping[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM];
FM_TYPE messages_pong[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM];

FM_TYPE h_node[MAX_NODE][EMB_DIM];
