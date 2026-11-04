#ifndef __CONV_LAYER_H__
#define __CONV_LAYER_H__

#include "dcl.h"

void compute_CONV_layer(
    int layer_num,
    FM_TYPE message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_TYPE next_message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    int num_of_nodes
);

#endif
