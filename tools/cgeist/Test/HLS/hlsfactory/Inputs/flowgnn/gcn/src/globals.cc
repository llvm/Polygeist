#include "dcl.h"

WT_TYPE convs_weight[NUM_LAYERS][100][100];
WT_TYPE convs_bias[NUM_LAYERS][100];
WT_TYPE convs_root_emb_weight[NUM_LAYERS][100];

WT_TYPE bn_weight[NUM_LAYERS][100];
WT_TYPE bn_bias[NUM_LAYERS][100];
WT_TYPE bn_mean[NUM_LAYERS][100];
WT_TYPE bn_sqrt_var[NUM_LAYERS][100];

WT_TYPE edge_embedding_weights[EDGE_PARALLEL][NUM_LAYERS][ED_FEATURE_PER_LAYER][EMB_DIM];
WT_TYPE graph_pred_weights[NUM_TASK][EMB_DIM];
WT_TYPE graph_pred_bias[NUM_TASK];

int degree_table[MAX_NODE];
int degree_tables[EDGE_PARALLEL][MAX_NODE][2];
int neighbor_tables[EDGE_PARALLEL][MAX_EDGE];
WT_TYPE norms[EDGE_PARALLEL][MAX_EDGE];
edge_attr_t edge_attrs[EDGE_PARALLEL][MAX_EDGE];
int num_of_edges_per_pe[EDGE_PARALLEL];

// BRAM for intermediate storage
FM_TYPE messages_ping[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM];
FM_TYPE messages_pong[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM];

FM_TYPE h_node[MAX_NODE][EMB_DIM];
