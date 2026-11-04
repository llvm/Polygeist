#ifndef __TESTBENCH_H__
#define __TESTBENCH_H__

#include "../src/dcl.h"

constexpr int NUM_GRAPHS = 1;

extern WT_TYPE (*node_embedding_weight_fixed)[ND_FEATURE_TOTAL][EMB_DIM];
extern WT_TYPE (*edge_embedding_weight_fixed)[NUM_LAYERS][ED_FEATURE_PER_LAYER][EMB_DIM];

extern WT_TYPE (*convs_weight_fixed)[NUM_LAYERS][EMB_DIM][EMB_DIM];
extern WT_TYPE (*convs_bias_fixed)[NUM_LAYERS][EMB_DIM];
extern WT_TYPE (*convs_root_emb_weight_fixed)[NUM_LAYERS][EMB_DIM];

extern WT_TYPE (*bn_weight_fixed)[NUM_LAYERS][EMB_DIM];
extern WT_TYPE (*bn_bias_fixed)[NUM_LAYERS][EMB_DIM];
extern WT_TYPE (*bn_mean_fixed)[NUM_LAYERS][EMB_DIM];
extern WT_TYPE (*bn_var_fixed)[NUM_LAYERS][EMB_DIM];

extern WT_TYPE (*graph_pred_linear_weight_fixed)[NUM_TASK][EMB_DIM];
extern WT_TYPE (*graph_pred_linear_bias_fixed)[NUM_TASK];

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
