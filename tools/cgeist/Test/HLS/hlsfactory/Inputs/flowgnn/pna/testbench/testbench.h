#ifndef __TESTBENCH_H__
#define __TESTBENCH_H__

#include "../src/dcl.h"

constexpr int NUM_GRAPHS = 1;

extern WT_TYPE node_embedding_weight_fixed[ND_FEATURE_TOTAL][EMB_DIM];
extern WT_TYPE node_conv_weights_fixed[NUM_LAYERS][EMB_DIM][NUM_SCALERS][NUM_AGGRS][EMB_DIM];
extern WT_TYPE node_conv_bias_fixed[NUM_LAYERS][EMB_DIM];
extern WT_TYPE graph_mlp_1_weights_fixed[GRAPH_MLP_1_OUT][EMB_DIM];
extern WT_TYPE graph_mlp_1_bias_fixed[GRAPH_MLP_1_OUT];
extern WT_TYPE graph_mlp_2_weights_fixed[GRAPH_MLP_2_OUT][GRAPH_MLP_1_OUT];
extern WT_TYPE graph_mlp_2_bias_fixed[GRAPH_MLP_2_OUT];
extern WT_TYPE graph_mlp_3_weights_fixed[NUM_TASK][GRAPH_MLP_2_OUT];
extern WT_TYPE graph_mlp_3_bias_fixed[NUM_TASK];
extern WT_TYPE avg_deg_fixed;

void load_weights();
void fetch_one_graph(
    int g,
    char* graph_name,
    node_feature_t* node_feature,
    edge_t* edge_list,
    int num_of_nodes,
    int num_of_edges
);

#endif
