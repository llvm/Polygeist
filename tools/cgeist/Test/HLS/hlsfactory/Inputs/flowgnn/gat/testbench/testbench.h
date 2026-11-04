#ifndef __TESTBENCH_H__
#define __TESTBENCH_H__

#include "../src/dcl.h"

constexpr int NUM_GRAPHS = 1;

extern WT_TYPE scoring_fn_target_fixed[NUM_LAYERS][NUM_HEADS][EMB_DIM];
extern WT_TYPE scoring_fn_source_fixed[NUM_LAYERS][NUM_HEADS][EMB_DIM];
extern WT_TYPE linear_proj_weights_fixed[NUM_LAYERS][NUM_HEADS][EMB_DIM][NUM_HEADS][EMB_DIM];
extern WT_TYPE skip_proj_weights_fixed[NUM_LAYERS][NUM_HEADS][EMB_DIM][NUM_HEADS][EMB_DIM];
extern WT_TYPE graph_pred_weights_fixed[NUM_TASK][EMB_DIM];
extern WT_TYPE graph_pred_bias_fixed[NUM_TASK];

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
