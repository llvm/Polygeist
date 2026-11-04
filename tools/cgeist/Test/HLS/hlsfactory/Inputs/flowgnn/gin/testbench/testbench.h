#ifndef __TESTBENCH_H__
#define __TESTBENCH_H__

#include "../src/dcl.h"

constexpr int NUM_GRAPHS = 1;

extern WT_TYPE (*node_mlp_1_weights_fixed)[NUM_LAYERS][MLP_1_OUT][EMB_DIM];
extern WT_TYPE (*node_mlp_1_bias_fixed)[NUM_LAYERS][MLP_1_OUT];
extern WT_TYPE (*node_mlp_2_weights_fixed)[NUM_LAYERS][EMB_DIM][MLP_1_OUT];
extern WT_TYPE (*node_mlp_2_bias_fixed)[NUM_LAYERS][EMB_DIM];
extern WT_TYPE (*node_embedding_table_fixed)[ND_FEATURE_TOTAL][EMB_DIM];
extern WT_TYPE (*edge_embedding_table_fixed)[NUM_LAYERS][ED_FEATURE_PER_LAYER][EMB_DIM];
extern WT_TYPE (*graph_pred_linear_weight_fixed)[NUM_TASK][EMB_DIM];
extern WT_TYPE (*graph_pred_linear_bias_fixed)[NUM_TASK];
extern WT_TYPE (*eps_fixed)[NUM_LAYERS];

void load_weights();
void fetch_one_graph(
    int g,
    char* graph_name,
    node_feature_t* node_feature,
    edge_t* edge_list,
    edge_attr_t* edge_attr,
    int num_of_nodes,
    int num_of_edges
);

#endif
