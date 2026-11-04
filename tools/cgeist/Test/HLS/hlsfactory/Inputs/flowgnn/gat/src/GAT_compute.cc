#include "dcl.h"
#include "load_inputs.h"
#include "conv_layer.h"
#include "finalize.h"

extern "C" {
void GAT_compute_graphs(
    int num_graphs,
    int* nums_of_nodes,
    int* nums_of_edges,
    int* reload_weights,
    FM_TYPE out[][NUM_TASK],
    node_feature_t* node_feature_in,
    edge_t* edge_list_in,
    WT_TYPE scoring_fn_target_in[][NUM_LAYERS][NUM_HEADS][EMB_DIM],
    WT_TYPE scoring_fn_source_in[][NUM_LAYERS][NUM_HEADS][EMB_DIM],
    WT_TYPE linear_proj_weights_in[][NUM_LAYERS][NUM_HEADS][EMB_DIM][NUM_HEADS][EMB_DIM],
    WT_TYPE skip_proj_weights_in[][NUM_LAYERS][NUM_HEADS][EMB_DIM][NUM_HEADS][EMB_DIM],
    WT_TYPE graph_pred_weights_in[][NUM_TASK][EMB_DIM],
    WT_TYPE graph_pred_bias_in[][NUM_TASK]
)
{
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS INTERFACE m_axi depth=(1) port=nums_of_nodes offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=nums_of_edges offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=reload_weights offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=out offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(MAX_NODE) port=node_feature_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(MAX_EDGE) port=edge_list_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=scoring_fn_target_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=scoring_fn_source_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=linear_proj_weights_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=skip_proj_weights_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graph_pred_weights_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graph_pred_bias_in offset=slave bundle=mem

#pragma HLS AGGREGATE variable=h_node_ping
#pragma HLS AGGREGATE variable=h_node_pong
#pragma HLS AGGREGATE variable=out_nodes_features_skip_concat_bias_ping
#pragma HLS AGGREGATE variable=out_nodes_features_skip_concat_bias_pong
#pragma HLS AGGREGATE variable=scores_source_ping
#pragma HLS AGGREGATE variable=scores_source_pong
#pragma HLS AGGREGATE variable=scores_target_ping
#pragma HLS AGGREGATE variable=scores_target_pong

    for (int graph = 0, weights_ndx = -1, nodes_offset = 0, edges_offset = 0; graph < num_graphs; graph++)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_NUM_GRAPHS max=ANALYSIS_NUM_GRAPHS avg=ANALYSIS_NUM_GRAPHS
        int num_of_nodes = nums_of_nodes[graph];
        int num_of_edges = nums_of_edges[graph];
        bool reload_weights_graph = reload_weights[graph];

        if (reload_weights_graph)
        {
            weights_ndx++;
            load_weights(
                scoring_fn_target_in[weights_ndx],
                scoring_fn_source_in[weights_ndx],
                linear_proj_weights_in[weights_ndx],
                skip_proj_weights_in[weights_ndx],
                graph_pred_weights_in[weights_ndx],
                graph_pred_bias_in[weights_ndx]
            );
        }

        load_graph(
            &edge_list_in[edges_offset],
            num_of_nodes,
            num_of_edges
        );
        load_input_node_embeddings(node_feature_in, num_of_nodes);

        for (int i = 0; i < NUM_LAYERS; i++)
        {
            if (i % 2 == 0)
                compute_CONV_layer(
                    i,
                    h_node_ping,
                    h_node_pong,
                    scores_source_ping,
                    scores_source_pong,
                    scores_target_ping,
                    scores_target_pong,
                    out_nodes_features_skip_concat_bias_ping,
                    out_nodes_features_skip_concat_bias_pong,
                    out[graph],
                    num_of_nodes
                );
            else
                compute_CONV_layer(
                    i,
                    h_node_pong,
                    h_node_ping,
                    scores_source_pong,
                    scores_source_ping,
                    scores_target_pong,
                    scores_target_ping,
                    out_nodes_features_skip_concat_bias_pong,
                    out_nodes_features_skip_concat_bias_ping,
                    out[graph],
                    num_of_nodes
                );
        }

        nodes_offset += num_of_nodes;
        edges_offset += num_of_edges;
    }
}
}
