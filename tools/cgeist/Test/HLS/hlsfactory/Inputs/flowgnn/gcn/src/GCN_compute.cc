#include "dcl.h"
#include "load_inputs.h"
#include "conv_layer.h"
#include "finalize.h"

extern "C" {
void GCN_compute_graphs(
    int num_graphs,
    int* nums_of_nodes,
    int* nums_of_edges,
    int* reload_weights,
    FM_TYPE* out,
    node_feature_t* node_feature_in,
    edge_t* edge_list_in,
    edge_attr_t* edge_attr_in,
    WT_TYPE node_embedding_weight_in[][ND_FEATURE_TOTAL][EMB_DIM],
    WT_TYPE edge_embedding_weight_in[][NUM_LAYERS][ED_FEATURE_PER_LAYER][EMB_DIM],
    WT_TYPE convs_weight_in[][NUM_LAYERS][100][100],
    WT_TYPE convs_bias_in[][NUM_LAYERS][100],
    WT_TYPE convs_root_emb_weight_in[][NUM_LAYERS][100],
    WT_TYPE bn_weight_in[][NUM_LAYERS][100],
    WT_TYPE bn_bias_in[][NUM_LAYERS][100],
    WT_TYPE bn_mean_in[][NUM_LAYERS][100],
    WT_TYPE bn_var_in[][NUM_LAYERS][100],
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
#pragma HLS INTERFACE m_axi depth=(MAX_EDGE) port=edge_attr_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=node_embedding_weight_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=edge_embedding_weight_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=convs_weight_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=convs_bias_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=convs_root_emb_weight_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=bn_weight_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=bn_bias_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=bn_mean_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=bn_var_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graph_pred_weights_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graph_pred_bias_in offset=slave bundle=mem

#pragma HLS bind_storage variable=degree_table type=RAM_2P impl=uram
#pragma HLS bind_storage variable=degree_tables type=RAM_2P impl=uram
#pragma HLS bind_storage variable=neighbor_tables type=RAM_2P impl=uram
#pragma HLS bind_storage variable=norms type=RAM_2P impl=uram
#pragma HLS bind_storage variable=edge_attrs type=RAM_2P impl=uram
#pragma HLS bind_storage variable=messages_ping type=RAM_2P impl=uram
#pragma HLS bind_storage variable=messages_pong type=RAM_2P impl=uram
#pragma HLS bind_storage variable=h_node type=RAM_2P impl=uram

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
                convs_weight_in[weights_ndx],
                convs_bias_in[weights_ndx],
                convs_root_emb_weight_in[weights_ndx],
                bn_weight_in[weights_ndx],
                bn_bias_in[weights_ndx],
                bn_mean_in[weights_ndx],
                bn_var_in[weights_ndx],
                edge_embedding_weight_in[weights_ndx],
                graph_pred_weights_in[weights_ndx],
                graph_pred_bias_in[weights_ndx]
            );
        }

        load_graph(
            &edge_list_in[edges_offset],
            &edge_attr_in[edges_offset],
            num_of_nodes,
            num_of_edges
        );
        load_input_node_embeddings(
            &node_feature_in[nodes_offset],
            node_embedding_weight_in[weights_ndx],
            messages_pong,
            num_of_nodes
        );

        for (int i = 0; i < NUM_LAYERS; i++)
        {
            if (i % 2 == 0)
                compute_CONV_layer(i, messages_ping, messages_pong, num_of_nodes);
            else
                compute_CONV_layer(i, messages_pong, messages_ping, num_of_nodes);
        }

        if (NUM_LAYERS % 2 == 0)
            finalize(h_node, messages_ping, graph_pred_weights, graph_pred_bias, &out[graph], num_of_nodes);
        else
            finalize(h_node, messages_pong, graph_pred_weights, graph_pred_bias, &out[graph], num_of_nodes);

        nodes_offset += num_of_nodes;
        edges_offset += num_of_edges;
    }
}
}
