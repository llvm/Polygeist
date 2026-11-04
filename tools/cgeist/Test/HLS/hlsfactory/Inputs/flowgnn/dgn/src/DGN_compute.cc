#include "dcl.h"
#include "load_inputs.h"
#include "conv_layer.h"

extern "C" {
void DGN_compute_graphs(
    int num_graphs,
    int* nums_of_nodes,
    int* nums_of_edges,
    int* reload_weights,
    FM_TYPE* out,
    node_feature_t* node_feature_in,
    node_eigen_t* node_eigen_in,
    edge_t* edge_list_in,
    WT_TYPE embedding_h_atom_embedding_list_weights_in[][9][119][100],
    WT_TYPE layers_posttrans_fully_connected_0_linear_weight_in[][4][100][200],
    WT_TYPE layers_posttrans_fully_connected_0_linear_bias_in[][4][100],
    WT_TYPE MLP_layer_FC_layers_0_weight_in[][50][100],
    WT_TYPE MLP_layer_FC_layers_0_bias_in[][50],
    WT_TYPE MLP_layer_FC_layers_1_weight_in[][25][50],
    WT_TYPE MLP_layer_FC_layers_1_bias_in[][25],
    WT_TYPE MLP_layer_FC_layers_2_weight_in[][1][25],
    WT_TYPE MLP_layer_FC_layers_2_bias_in[][1]
)
{
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS INTERFACE m_axi depth=(1) port=nums_of_nodes offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=nums_of_edges offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=reload_weights offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=out offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(MAX_NODE) port=node_feature_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(MAX_NODE) port=node_eigen_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(MAX_EDGE) port=edge_list_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=embedding_h_atom_embedding_list_weights_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=layers_posttrans_fully_connected_0_linear_weight_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=layers_posttrans_fully_connected_0_linear_bias_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=MLP_layer_FC_layers_0_weight_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=MLP_layer_FC_layers_0_bias_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=MLP_layer_FC_layers_1_weight_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=MLP_layer_FC_layers_1_bias_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=MLP_layer_FC_layers_2_weight_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=MLP_layer_FC_layers_2_bias_in offset=slave bundle=mem

#pragma HLS bind_storage variable=layers_posttrans_fully_connected_0_linear_weight type=RAM_2P impl=bram
#pragma HLS bind_storage variable=layers_posttrans_fully_connected_0_linear_bias type=RAM_2P impl=bram

#pragma HLS bind_storage variable=degree_table type=RAM_2P impl=uram
#pragma HLS bind_storage variable=degree_tables type=RAM_2P impl=uram
#pragma HLS bind_storage variable=neighbor_tables type=RAM_2P impl=uram
#pragma HLS bind_storage variable=eig_w type=RAM_2P impl=uram
#pragma HLS bind_storage variable=eig_abssums type=RAM_2P impl=uram
#pragma HLS bind_storage variable=eigw_sums type=RAM_2P impl=uram
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
                layers_posttrans_fully_connected_0_linear_weight_in[weights_ndx],
                layers_posttrans_fully_connected_0_linear_bias_in[weights_ndx],
                MLP_layer_FC_layers_0_weight_in[weights_ndx],
                MLP_layer_FC_layers_0_bias_in[weights_ndx],
                MLP_layer_FC_layers_1_weight_in[weights_ndx],
                MLP_layer_FC_layers_1_bias_in[weights_ndx],
                MLP_layer_FC_layers_2_weight_in[weights_ndx],
                MLP_layer_FC_layers_2_bias_in[weights_ndx]
            );
        }

        load_graph(
            &edge_list_in[edges_offset],
            &node_eigen_in[nodes_offset],
            num_of_nodes,
            num_of_edges
        );

        for (int i = 0; i <= NUM_LAYERS; i++)
        {
            if (i % 2 == 0)
                compute_CONV_layer(
                    i,
                    messages_ping,
                    messages_pong,
                    &node_feature_in[nodes_offset],
                    embedding_h_atom_embedding_list_weights_in[weights_ndx],
                    &out[graph],
                    num_of_nodes
                );
            else
                compute_CONV_layer(
                    i,
                    messages_pong,
                    messages_ping,
                    &node_feature_in[nodes_offset],
                    embedding_h_atom_embedding_list_weights_in[weights_ndx],
                    &out[graph],
                    num_of_nodes
                );
        }

        nodes_offset += num_of_nodes;
        edges_offset += num_of_edges;
    }
}
}
