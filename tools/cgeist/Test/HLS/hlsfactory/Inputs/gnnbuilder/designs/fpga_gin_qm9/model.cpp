
#include "model.h"

// Model Parameters //
W_TYPE mlp_head_linear_layers_0_weight_fixed[64][192];
W_TYPE mlp_head_linear_layers_0_bias_fixed[64];
W_TYPE mlp_head_linear_layers_1_weight_fixed[64][64];
W_TYPE mlp_head_linear_layers_1_bias_fixed[64];
W_TYPE mlp_head_linear_layers_2_weight_fixed[64][64];
W_TYPE mlp_head_linear_layers_2_bias_fixed[64];
W_TYPE mlp_head_linear_layers_3_weight_fixed[64][64];
W_TYPE mlp_head_linear_layers_3_bias_fixed[64];
W_TYPE mlp_head_linear_layers_4_weight_fixed[19][64];
W_TYPE mlp_head_linear_layers_4_bias_fixed[19];
W_TYPE gnn_convs_0_mlp_linear_0_weight_fixed[64][11];
W_TYPE gnn_convs_0_mlp_linear_0_bias_fixed[64];
W_TYPE gnn_convs_0_mlp_linear_1_weight_fixed[128][64];
W_TYPE gnn_convs_0_mlp_linear_1_bias_fixed[128];
W_TYPE gnn_convs_1_mlp_linear_0_weight_fixed[64][128];
W_TYPE gnn_convs_1_mlp_linear_0_bias_fixed[64];
W_TYPE gnn_convs_1_mlp_linear_1_weight_fixed[128][64];
W_TYPE gnn_convs_1_mlp_linear_1_bias_fixed[128];
W_TYPE gnn_convs_2_mlp_linear_0_weight_fixed[64][128];
W_TYPE gnn_convs_2_mlp_linear_0_bias_fixed[64];
W_TYPE gnn_convs_2_mlp_linear_1_weight_fixed[128][64];
W_TYPE gnn_convs_2_mlp_linear_1_bias_fixed[128];
W_TYPE gnn_convs_3_mlp_linear_0_weight_fixed[64][128];
W_TYPE gnn_convs_3_mlp_linear_0_bias_fixed[64];
W_TYPE gnn_convs_3_mlp_linear_1_weight_fixed[128][64];
W_TYPE gnn_convs_3_mlp_linear_1_bias_fixed[128];
W_TYPE gnn_convs_4_mlp_linear_0_weight_fixed[64][128];
W_TYPE gnn_convs_4_mlp_linear_0_bias_fixed[64];
W_TYPE gnn_convs_4_mlp_linear_1_weight_fixed[128][64];
W_TYPE gnn_convs_4_mlp_linear_1_bias_fixed[128];
W_TYPE gnn_convs_5_mlp_linear_0_weight_fixed[64][128];
W_TYPE gnn_convs_5_mlp_linear_0_bias_fixed[64];
W_TYPE gnn_convs_5_mlp_linear_1_weight_fixed[64][64];
W_TYPE gnn_convs_5_mlp_linear_1_bias_fixed[64];

// Input Graph Data // 
F_TYPE node_feature_table[600][11];
int edge_list[600][2];

// Neighbor Table Data //
int neightbor_table_offsets[600];
int neighbor_table[600];

// In-Degree and Out-Degree Table //
int in_degree_table[600];
int out_degree_table[600];

F_TYPE node_emb_out[600][64] = {0};




void compute_gnn_head(int n_nodes, int n_edges){
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW
    


    // activations

    
    // generate an input and output buffer for each conv layer

    static F_TYPE conv_0_in[600][11];
    static F_TYPE conv_0_in_conv[600][11];
    static F_TYPE conv_0_in_skip[600][11];
    static F_TYPE conv_0_post_conv[600][128];
    static F_TYPE conv_0_post_activation[600][128];
    static F_TYPE conv_0_post_skip_connection[600][128];
    static F_TYPE conv_0_out[600][128];
    
    static int edge_list_0[600][2];
    static int neightbor_table_offsets_0[600];
    static int neighbor_table_0[600];
    static int in_degree_table_0[600];
    static int out_degree_table_0[600];

    #pragma HLS stable variable=edge_list_0
    #pragma HLS stable variable=neightbor_table_offsets_0
    #pragma HLS stable variable=neighbor_table_0
    #pragma HLS stable variable=in_degree_table_0
    #pragma HLS stable variable=out_degree_table_0
    // generate an input and output buffer for each conv layer

    static F_TYPE conv_1_in[600][128];
    static F_TYPE conv_1_in_conv[600][128];
    static F_TYPE conv_1_in_skip[600][128];
    static F_TYPE conv_1_post_conv[600][128];
    static F_TYPE conv_1_post_activation[600][128];
    static F_TYPE conv_1_post_skip_connection[600][128];
    static F_TYPE conv_1_out[600][128];
    
    static int edge_list_1[600][2];
    static int neightbor_table_offsets_1[600];
    static int neighbor_table_1[600];
    static int in_degree_table_1[600];
    static int out_degree_table_1[600];

    #pragma HLS stable variable=edge_list_1
    #pragma HLS stable variable=neightbor_table_offsets_1
    #pragma HLS stable variable=neighbor_table_1
    #pragma HLS stable variable=in_degree_table_1
    #pragma HLS stable variable=out_degree_table_1
    // generate an input and output buffer for each conv layer

    static F_TYPE conv_2_in[600][128];
    static F_TYPE conv_2_in_conv[600][128];
    static F_TYPE conv_2_in_skip[600][128];
    static F_TYPE conv_2_post_conv[600][128];
    static F_TYPE conv_2_post_activation[600][128];
    static F_TYPE conv_2_post_skip_connection[600][128];
    static F_TYPE conv_2_out[600][128];
    
    static int edge_list_2[600][2];
    static int neightbor_table_offsets_2[600];
    static int neighbor_table_2[600];
    static int in_degree_table_2[600];
    static int out_degree_table_2[600];

    #pragma HLS stable variable=edge_list_2
    #pragma HLS stable variable=neightbor_table_offsets_2
    #pragma HLS stable variable=neighbor_table_2
    #pragma HLS stable variable=in_degree_table_2
    #pragma HLS stable variable=out_degree_table_2
    // generate an input and output buffer for each conv layer

    static F_TYPE conv_3_in[600][128];
    static F_TYPE conv_3_in_conv[600][128];
    static F_TYPE conv_3_in_skip[600][128];
    static F_TYPE conv_3_post_conv[600][128];
    static F_TYPE conv_3_post_activation[600][128];
    static F_TYPE conv_3_post_skip_connection[600][128];
    static F_TYPE conv_3_out[600][128];
    
    static int edge_list_3[600][2];
    static int neightbor_table_offsets_3[600];
    static int neighbor_table_3[600];
    static int in_degree_table_3[600];
    static int out_degree_table_3[600];

    #pragma HLS stable variable=edge_list_3
    #pragma HLS stable variable=neightbor_table_offsets_3
    #pragma HLS stable variable=neighbor_table_3
    #pragma HLS stable variable=in_degree_table_3
    #pragma HLS stable variable=out_degree_table_3
    // generate an input and output buffer for each conv layer

    static F_TYPE conv_4_in[600][128];
    static F_TYPE conv_4_in_conv[600][128];
    static F_TYPE conv_4_in_skip[600][128];
    static F_TYPE conv_4_post_conv[600][128];
    static F_TYPE conv_4_post_activation[600][128];
    static F_TYPE conv_4_post_skip_connection[600][128];
    static F_TYPE conv_4_out[600][128];
    
    static int edge_list_4[600][2];
    static int neightbor_table_offsets_4[600];
    static int neighbor_table_4[600];
    static int in_degree_table_4[600];
    static int out_degree_table_4[600];

    #pragma HLS stable variable=edge_list_4
    #pragma HLS stable variable=neightbor_table_offsets_4
    #pragma HLS stable variable=neighbor_table_4
    #pragma HLS stable variable=in_degree_table_4
    #pragma HLS stable variable=out_degree_table_4
    // generate an input and output buffer for each conv layer

    static F_TYPE conv_5_in[600][128];
    static F_TYPE conv_5_in_conv[600][128];
    static F_TYPE conv_5_in_skip[600][128];
    static F_TYPE conv_5_post_conv[600][64];
    static F_TYPE conv_5_post_activation[600][64];
    static F_TYPE conv_5_post_skip_connection[600][64];
    static F_TYPE conv_5_out[600][64];
    
    static int edge_list_5[600][2];
    static int neightbor_table_offsets_5[600];
    static int neighbor_table_5[600];
    static int in_degree_table_5[600];
    static int out_degree_table_5[600];

    #pragma HLS stable variable=edge_list_5
    #pragma HLS stable variable=neightbor_table_offsets_5
    #pragma HLS stable variable=neighbor_table_5
    #pragma HLS stable variable=in_degree_table_5
    #pragma HLS stable variable=out_degree_table_5

    for (int i = 0; i < n_edges; i++) {
        #pragma HLS loop_tripcount min=0 max=24
        for (int j = 0; j < 2; j++) {
            int val = edge_list[i][j];
            edge_list_0[i][j] = val;
            edge_list_1[i][j] = val;
            edge_list_2[i][j] = val;
            edge_list_3[i][j] = val;
            edge_list_4[i][j] = val;
            edge_list_5[i][j] = val;
        }
    }
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        int val = neightbor_table_offsets[i];
        neightbor_table_offsets_0[i] = val;
        neightbor_table_offsets_1[i] = val;
        neightbor_table_offsets_2[i] = val;
        neightbor_table_offsets_3[i] = val;
        neightbor_table_offsets_4[i] = val;
        neightbor_table_offsets_5[i] = val;
    }
    for (int i = 0; i < n_edges; i++) {
        #pragma HLS loop_tripcount min=0 max=24
        int val = neighbor_table[i];
        neighbor_table_0[i] = val;
        neighbor_table_1[i] = val;
        neighbor_table_2[i] = val;
        neighbor_table_3[i] = val;
        neighbor_table_4[i] = val;
        neighbor_table_5[i] = val;
    }
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        int val = in_degree_table[i];
        in_degree_table_0[i] = val;
        in_degree_table_1[i] = val;
        in_degree_table_2[i] = val;
        in_degree_table_3[i] = val;
        in_degree_table_4[i] = val;
        in_degree_table_5[i] = val;
    }
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        int val = out_degree_table[i];
        out_degree_table_0[i] = val;
        out_degree_table_1[i] = val;
        out_degree_table_2[i] = val;
        out_degree_table_3[i] = val;
        out_degree_table_4[i] = val;
        out_degree_table_5[i] = val;
    }

    // copy node features to the first input buffer
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 11; j++) {
            conv_0_in[i][j] = node_feature_table[i][j];
        }
    }

    ///////////////////////////////
    // conv layer 0
    ///////////////////////////////

    // do_skip = False

    // not_last = True

    // read to input buffer and skip connection buffer if skip connection is enabled
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 11; j++) {
            conv_0_in_conv[i][j] = conv_0_in[i][j];
        }
    }
    // compute the conv layer
    // placeholder
    
    gin_conv<
        600,
        600,
        11,
        128,
        64,
        F_TYPE,
        12,
        24,
        1,
        1,
        8
    >(
        n_nodes,
        n_edges,
        conv_0_in_conv,
        conv_0_post_conv,
        edge_list_0,
        neightbor_table_offsets_0,
        neighbor_table_0,
        in_degree_table_0,
        out_degree_table_0,
        gnn_convs_0_mlp_linear_0_weight_fixed,
        gnn_convs_0_mlp_linear_0_bias_fixed,
        gnn_convs_0_mlp_linear_1_weight_fixed,
        gnn_convs_0_mlp_linear_1_bias_fixed,
        W_TYPE(0.0)
    );

    // skip connection
    // activation
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 128; j++) {
                conv_0_post_activation[i][j] = activation_relu(conv_0_post_conv[i][j]);
        }
    }
    // copy output to conv_out
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 128; j++) {
            conv_0_out[i][j] = conv_0_post_activation[i][j];
        }
    }

    ///////////////////////////////

    // copy conv output to next conv input: conv_0_out -> conv_1_in
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 128; j++) {
            conv_1_in[i][j] = conv_0_out[i][j];
        }
    }

    ///////////////////////////////

    ///////////////////////////////
    // conv layer 1
    ///////////////////////////////

    // do_skip = True

    // not_last = True

    // read to input buffer and skip connection buffer if skip connection is enabled
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 128; j++) {
            conv_1_in_conv[i][j] = conv_1_in[i][j];
                conv_1_in_skip[i][j] = conv_1_in[i][j];
        }
    }
    // compute the conv layer
    // placeholder
    
    gin_conv<
        600,
        600,
        128,
        128,
        64,
        F_TYPE,
        12,
        24,
        1,
        8,
        8
    >(
        n_nodes,
        n_edges,
        conv_1_in_conv,
        conv_1_post_conv,
        edge_list_1,
        neightbor_table_offsets_1,
        neighbor_table_1,
        in_degree_table_1,
        out_degree_table_1,
        gnn_convs_1_mlp_linear_0_weight_fixed,
        gnn_convs_1_mlp_linear_0_bias_fixed,
        gnn_convs_1_mlp_linear_1_weight_fixed,
        gnn_convs_1_mlp_linear_1_bias_fixed,
        W_TYPE(0.0)
    );

    // skip connection
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 128; j++) {
            conv_1_post_skip_connection[i][j] = conv_1_in_skip[i][j] + conv_1_post_conv[i][j];
        }
    }
    // activation
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 128; j++) {
                conv_1_post_activation[i][j] = activation_relu(conv_1_post_skip_connection[i][j]);
        }
    }
    // copy output to conv_out
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 128; j++) {
            conv_1_out[i][j] = conv_1_post_activation[i][j];
        }
    }

    ///////////////////////////////

    // copy conv output to next conv input: conv_1_out -> conv_2_in
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 128; j++) {
            conv_2_in[i][j] = conv_1_out[i][j];
        }
    }

    ///////////////////////////////

    ///////////////////////////////
    // conv layer 2
    ///////////////////////////////

    // do_skip = True

    // not_last = True

    // read to input buffer and skip connection buffer if skip connection is enabled
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 128; j++) {
            conv_2_in_conv[i][j] = conv_2_in[i][j];
                conv_2_in_skip[i][j] = conv_2_in[i][j];
        }
    }
    // compute the conv layer
    // placeholder
    
    gin_conv<
        600,
        600,
        128,
        128,
        64,
        F_TYPE,
        12,
        24,
        1,
        8,
        8
    >(
        n_nodes,
        n_edges,
        conv_2_in_conv,
        conv_2_post_conv,
        edge_list_2,
        neightbor_table_offsets_2,
        neighbor_table_2,
        in_degree_table_2,
        out_degree_table_2,
        gnn_convs_2_mlp_linear_0_weight_fixed,
        gnn_convs_2_mlp_linear_0_bias_fixed,
        gnn_convs_2_mlp_linear_1_weight_fixed,
        gnn_convs_2_mlp_linear_1_bias_fixed,
        W_TYPE(0.0)
    );

    // skip connection
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 128; j++) {
            conv_2_post_skip_connection[i][j] = conv_2_in_skip[i][j] + conv_2_post_conv[i][j];
        }
    }
    // activation
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 128; j++) {
                conv_2_post_activation[i][j] = activation_relu(conv_2_post_skip_connection[i][j]);
        }
    }
    // copy output to conv_out
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 128; j++) {
            conv_2_out[i][j] = conv_2_post_activation[i][j];
        }
    }

    ///////////////////////////////

    // copy conv output to next conv input: conv_2_out -> conv_3_in
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 128; j++) {
            conv_3_in[i][j] = conv_2_out[i][j];
        }
    }

    ///////////////////////////////

    ///////////////////////////////
    // conv layer 3
    ///////////////////////////////

    // do_skip = True

    // not_last = True

    // read to input buffer and skip connection buffer if skip connection is enabled
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 128; j++) {
            conv_3_in_conv[i][j] = conv_3_in[i][j];
                conv_3_in_skip[i][j] = conv_3_in[i][j];
        }
    }
    // compute the conv layer
    // placeholder
    
    gin_conv<
        600,
        600,
        128,
        128,
        64,
        F_TYPE,
        12,
        24,
        1,
        8,
        8
    >(
        n_nodes,
        n_edges,
        conv_3_in_conv,
        conv_3_post_conv,
        edge_list_3,
        neightbor_table_offsets_3,
        neighbor_table_3,
        in_degree_table_3,
        out_degree_table_3,
        gnn_convs_3_mlp_linear_0_weight_fixed,
        gnn_convs_3_mlp_linear_0_bias_fixed,
        gnn_convs_3_mlp_linear_1_weight_fixed,
        gnn_convs_3_mlp_linear_1_bias_fixed,
        W_TYPE(0.0)
    );

    // skip connection
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 128; j++) {
            conv_3_post_skip_connection[i][j] = conv_3_in_skip[i][j] + conv_3_post_conv[i][j];
        }
    }
    // activation
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 128; j++) {
                conv_3_post_activation[i][j] = activation_relu(conv_3_post_skip_connection[i][j]);
        }
    }
    // copy output to conv_out
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 128; j++) {
            conv_3_out[i][j] = conv_3_post_activation[i][j];
        }
    }

    ///////////////////////////////

    // copy conv output to next conv input: conv_3_out -> conv_4_in
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 128; j++) {
            conv_4_in[i][j] = conv_3_out[i][j];
        }
    }

    ///////////////////////////////

    ///////////////////////////////
    // conv layer 4
    ///////////////////////////////

    // do_skip = True

    // not_last = True

    // read to input buffer and skip connection buffer if skip connection is enabled
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 128; j++) {
            conv_4_in_conv[i][j] = conv_4_in[i][j];
                conv_4_in_skip[i][j] = conv_4_in[i][j];
        }
    }
    // compute the conv layer
    // placeholder
    
    gin_conv<
        600,
        600,
        128,
        128,
        64,
        F_TYPE,
        12,
        24,
        1,
        8,
        8
    >(
        n_nodes,
        n_edges,
        conv_4_in_conv,
        conv_4_post_conv,
        edge_list_4,
        neightbor_table_offsets_4,
        neighbor_table_4,
        in_degree_table_4,
        out_degree_table_4,
        gnn_convs_4_mlp_linear_0_weight_fixed,
        gnn_convs_4_mlp_linear_0_bias_fixed,
        gnn_convs_4_mlp_linear_1_weight_fixed,
        gnn_convs_4_mlp_linear_1_bias_fixed,
        W_TYPE(0.0)
    );

    // skip connection
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 128; j++) {
            conv_4_post_skip_connection[i][j] = conv_4_in_skip[i][j] + conv_4_post_conv[i][j];
        }
    }
    // activation
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 128; j++) {
                conv_4_post_activation[i][j] = activation_relu(conv_4_post_skip_connection[i][j]);
        }
    }
    // copy output to conv_out
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 128; j++) {
            conv_4_out[i][j] = conv_4_post_activation[i][j];
        }
    }

    ///////////////////////////////

    // copy conv output to next conv input: conv_4_out -> conv_5_in
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 128; j++) {
            conv_5_in[i][j] = conv_4_out[i][j];
        }
    }

    ///////////////////////////////

    ///////////////////////////////
    // conv layer 5
    ///////////////////////////////

    // do_skip = False

    // not_last = False

    // read to input buffer and skip connection buffer if skip connection is enabled
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 128; j++) {
            conv_5_in_conv[i][j] = conv_5_in[i][j];
        }
    }
    // compute the conv layer
    // placeholder
    
    gin_conv<
        600,
        600,
        128,
        64,
        64,
        F_TYPE,
        12,
        24,
        1,
        8,
        8
    >(
        n_nodes,
        n_edges,
        conv_5_in_conv,
        conv_5_post_conv,
        edge_list_5,
        neightbor_table_offsets_5,
        neighbor_table_5,
        in_degree_table_5,
        out_degree_table_5,
        gnn_convs_5_mlp_linear_0_weight_fixed,
        gnn_convs_5_mlp_linear_0_bias_fixed,
        gnn_convs_5_mlp_linear_1_weight_fixed,
        gnn_convs_5_mlp_linear_1_bias_fixed,
        W_TYPE(0.0)
    );

    // skip connection
    // activation
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 64; j++) {
                conv_5_post_activation[i][j] = activation_relu(conv_5_post_conv[i][j]);
        }
    }
    // copy output to conv_out
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 64; j++) {
            conv_5_out[i][j] = conv_5_post_activation[i][j];
        }
    }

    ///////////////////////////////


    ///////////////////////////////


    // copy last conv output to node_emb_out

    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 64; j++) {
            node_emb_out[i][j] = conv_5_out[i][j];
        }
    }



}


// macro for global pooling


F_TYPE global_pooling_buffer[192] = {0};

void compute_global_graph_pooling(int n_nodes, int n_edges){
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW
    #pragma HLS ARRAY_PARTITION variable=global_pooling_buffer block factor=3 dim=1
    
    static F_TYPE global_pooling_0_in[600][64];
    static F_TYPE global_pooling_0_out[64];
    static F_TYPE global_pooling_1_in[600][64];
    static F_TYPE global_pooling_1_out[64];
    static F_TYPE global_pooling_2_in[600][64];
    static F_TYPE global_pooling_2_out[64];

    // copy node_emb_out to global_pooling_in buffers
    for (int i = 0; i < n_nodes; i++) {
        #pragma HLS loop_tripcount min=0 max=12
        for (int j = 0; j < 64; j++) {
            F_TYPE node_emb_out_i_j = node_emb_out[i][j];
            global_pooling_0_in[i][j] = node_emb_out_i_j;
            global_pooling_1_in[i][j] = node_emb_out_i_j;
            global_pooling_2_in[i][j] = node_emb_out_i_j;
        }
    }

    global_add_pool<
        600,
        600,
        64,
        F_TYPE,
        12,
        24
    >(
        n_nodes,
        n_edges,
        global_pooling_0_in,
        global_pooling_0_out
    );

    global_mean_pool<
        600,
        600,
        64,
        F_TYPE,
        12,
        24
    >(
        n_nodes,
        n_edges,
        global_pooling_1_in,
        global_pooling_1_out
    );

    global_max_pool<
        600,
        600,
        64,
        F_TYPE,
        12,
        24
    >(
        n_nodes,
        n_edges,
        global_pooling_2_in,
        global_pooling_2_out
    );


    for (int i = 0; i < 64; i++) {
        global_pooling_buffer[0 * 64 + i] = global_pooling_0_out[i];
        global_pooling_buffer[1 * 64 + i] = global_pooling_1_out[i];
        global_pooling_buffer[2 * 64 + i] = global_pooling_2_out[i];
    }
}


F_TYPE mlp_buffer_out[19] = {0};

void compute_mlp_head(){
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW


    static F_TYPE mlp_0_in[192];
    static F_TYPE mlp_0_out_linear[64];
    static F_TYPE mlp_0_out_activation[64];
    static F_TYPE mlp_0_out[64];

    static F_TYPE mlp_1_in[64];
    static F_TYPE mlp_1_out_linear[64];
    static F_TYPE mlp_1_out_activation[64];
    static F_TYPE mlp_1_out[64];

    static F_TYPE mlp_2_in[64];
    static F_TYPE mlp_2_out_linear[64];
    static F_TYPE mlp_2_out_activation[64];
    static F_TYPE mlp_2_out[64];

    static F_TYPE mlp_3_in[64];
    static F_TYPE mlp_3_out_linear[64];
    static F_TYPE mlp_3_out_activation[64];
    static F_TYPE mlp_3_out[64];

    static F_TYPE mlp_4_in[64];
    static F_TYPE mlp_4_out_linear[19];
    static F_TYPE mlp_4_out_activation[19];
    static F_TYPE mlp_4_out[19];


    copy_1d<192, F_TYPE>(global_pooling_buffer, mlp_0_in);

    
    // linear layer 0
    // not_last: True

    linear_buffered<
        192,
        64,
        8,
        8,
        F_TYPE
    >(
        mlp_0_in,
        mlp_0_out_linear,
        mlp_head_linear_layers_0_weight_fixed,
        mlp_head_linear_layers_0_bias_fixed
    );

    for (int i = 0; i < 64; i++) {
        mlp_0_out_activation[i] = activation_relu(mlp_0_out_linear[i]);
    }

    copy_1d<64, F_TYPE>(mlp_0_out_activation, mlp_0_out);
    copy_1d<64, F_TYPE>(mlp_0_out, mlp_1_in);


    
    // linear layer 1
    // not_last: True

    linear_buffered<
        64,
        64,
        8,
        8,
        F_TYPE
    >(
        mlp_1_in,
        mlp_1_out_linear,
        mlp_head_linear_layers_1_weight_fixed,
        mlp_head_linear_layers_1_bias_fixed
    );

    for (int i = 0; i < 64; i++) {
        mlp_1_out_activation[i] = activation_relu(mlp_1_out_linear[i]);
    }

    copy_1d<64, F_TYPE>(mlp_1_out_activation, mlp_1_out);
    copy_1d<64, F_TYPE>(mlp_1_out, mlp_2_in);


    
    // linear layer 2
    // not_last: True

    linear_buffered<
        64,
        64,
        8,
        8,
        F_TYPE
    >(
        mlp_2_in,
        mlp_2_out_linear,
        mlp_head_linear_layers_2_weight_fixed,
        mlp_head_linear_layers_2_bias_fixed
    );

    for (int i = 0; i < 64; i++) {
        mlp_2_out_activation[i] = activation_relu(mlp_2_out_linear[i]);
    }

    copy_1d<64, F_TYPE>(mlp_2_out_activation, mlp_2_out);
    copy_1d<64, F_TYPE>(mlp_2_out, mlp_3_in);


    
    // linear layer 3
    // not_last: True

    linear_buffered<
        64,
        64,
        8,
        8,
        F_TYPE
    >(
        mlp_3_in,
        mlp_3_out_linear,
        mlp_head_linear_layers_3_weight_fixed,
        mlp_head_linear_layers_3_bias_fixed
    );

    for (int i = 0; i < 64; i++) {
        mlp_3_out_activation[i] = activation_relu(mlp_3_out_linear[i]);
    }

    copy_1d<64, F_TYPE>(mlp_3_out_activation, mlp_3_out);
    copy_1d<64, F_TYPE>(mlp_3_out, mlp_4_in);


    
    // linear layer 4
    // not_last: False

    linear_buffered<
        64,
        19,
        8,
        1,
        F_TYPE
    >(
        mlp_4_in,
        mlp_4_out_linear,
        mlp_head_linear_layers_4_weight_fixed,
        mlp_head_linear_layers_4_bias_fixed
    );

    copy_1d<19, F_TYPE>(mlp_4_out_linear, mlp_4_out);



    copy_1d<19, F_TYPE>(mlp_4_out, mlp_buffer_out);       
}


F_TYPE model_output_buffer[19] = {0};

void compute_model_output(){
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW


    copy_1d<19, F_TYPE>(mlp_buffer_out, model_output_buffer);
}

// Load Parameters Function //
void load_parameters(
    F_TYPE mlp_head_linear_layers_0_weight_fixed_in[64][192],
    F_TYPE mlp_head_linear_layers_0_bias_fixed_in[64],
    F_TYPE mlp_head_linear_layers_1_weight_fixed_in[64][64],
    F_TYPE mlp_head_linear_layers_1_bias_fixed_in[64],
    F_TYPE mlp_head_linear_layers_2_weight_fixed_in[64][64],
    F_TYPE mlp_head_linear_layers_2_bias_fixed_in[64],
    F_TYPE mlp_head_linear_layers_3_weight_fixed_in[64][64],
    F_TYPE mlp_head_linear_layers_3_bias_fixed_in[64],
    F_TYPE mlp_head_linear_layers_4_weight_fixed_in[19][64],
    F_TYPE mlp_head_linear_layers_4_bias_fixed_in[19],
    F_TYPE gnn_convs_0_mlp_linear_0_weight_fixed_in[64][11],
    F_TYPE gnn_convs_0_mlp_linear_0_bias_fixed_in[64],
    F_TYPE gnn_convs_0_mlp_linear_1_weight_fixed_in[128][64],
    F_TYPE gnn_convs_0_mlp_linear_1_bias_fixed_in[128],
    F_TYPE gnn_convs_1_mlp_linear_0_weight_fixed_in[64][128],
    F_TYPE gnn_convs_1_mlp_linear_0_bias_fixed_in[64],
    F_TYPE gnn_convs_1_mlp_linear_1_weight_fixed_in[128][64],
    F_TYPE gnn_convs_1_mlp_linear_1_bias_fixed_in[128],
    F_TYPE gnn_convs_2_mlp_linear_0_weight_fixed_in[64][128],
    F_TYPE gnn_convs_2_mlp_linear_0_bias_fixed_in[64],
    F_TYPE gnn_convs_2_mlp_linear_1_weight_fixed_in[128][64],
    F_TYPE gnn_convs_2_mlp_linear_1_bias_fixed_in[128],
    F_TYPE gnn_convs_3_mlp_linear_0_weight_fixed_in[64][128],
    F_TYPE gnn_convs_3_mlp_linear_0_bias_fixed_in[64],
    F_TYPE gnn_convs_3_mlp_linear_1_weight_fixed_in[128][64],
    F_TYPE gnn_convs_3_mlp_linear_1_bias_fixed_in[128],
    F_TYPE gnn_convs_4_mlp_linear_0_weight_fixed_in[64][128],
    F_TYPE gnn_convs_4_mlp_linear_0_bias_fixed_in[64],
    F_TYPE gnn_convs_4_mlp_linear_1_weight_fixed_in[128][64],
    F_TYPE gnn_convs_4_mlp_linear_1_bias_fixed_in[128],
    F_TYPE gnn_convs_5_mlp_linear_0_weight_fixed_in[64][128],
    F_TYPE gnn_convs_5_mlp_linear_0_bias_fixed_in[64],
    F_TYPE gnn_convs_5_mlp_linear_1_weight_fixed_in[64][64],
    F_TYPE gnn_convs_5_mlp_linear_1_bias_fixed_in[64]
){
    #pragma HLS INLINE off
    
    copy_2d<64, 192>(mlp_head_linear_layers_0_weight_fixed_in, mlp_head_linear_layers_0_weight_fixed);
    copy_1d<64>(mlp_head_linear_layers_0_bias_fixed_in, mlp_head_linear_layers_0_bias_fixed);
    copy_2d<64, 64>(mlp_head_linear_layers_1_weight_fixed_in, mlp_head_linear_layers_1_weight_fixed);
    copy_1d<64>(mlp_head_linear_layers_1_bias_fixed_in, mlp_head_linear_layers_1_bias_fixed);
    copy_2d<64, 64>(mlp_head_linear_layers_2_weight_fixed_in, mlp_head_linear_layers_2_weight_fixed);
    copy_1d<64>(mlp_head_linear_layers_2_bias_fixed_in, mlp_head_linear_layers_2_bias_fixed);
    copy_2d<64, 64>(mlp_head_linear_layers_3_weight_fixed_in, mlp_head_linear_layers_3_weight_fixed);
    copy_1d<64>(mlp_head_linear_layers_3_bias_fixed_in, mlp_head_linear_layers_3_bias_fixed);
    copy_2d<19, 64>(mlp_head_linear_layers_4_weight_fixed_in, mlp_head_linear_layers_4_weight_fixed);
    copy_1d<19>(mlp_head_linear_layers_4_bias_fixed_in, mlp_head_linear_layers_4_bias_fixed);
    copy_2d<64, 11>(gnn_convs_0_mlp_linear_0_weight_fixed_in, gnn_convs_0_mlp_linear_0_weight_fixed);
    copy_1d<64>(gnn_convs_0_mlp_linear_0_bias_fixed_in, gnn_convs_0_mlp_linear_0_bias_fixed);
    copy_2d<128, 64>(gnn_convs_0_mlp_linear_1_weight_fixed_in, gnn_convs_0_mlp_linear_1_weight_fixed);
    copy_1d<128>(gnn_convs_0_mlp_linear_1_bias_fixed_in, gnn_convs_0_mlp_linear_1_bias_fixed);
    copy_2d<64, 128>(gnn_convs_1_mlp_linear_0_weight_fixed_in, gnn_convs_1_mlp_linear_0_weight_fixed);
    copy_1d<64>(gnn_convs_1_mlp_linear_0_bias_fixed_in, gnn_convs_1_mlp_linear_0_bias_fixed);
    copy_2d<128, 64>(gnn_convs_1_mlp_linear_1_weight_fixed_in, gnn_convs_1_mlp_linear_1_weight_fixed);
    copy_1d<128>(gnn_convs_1_mlp_linear_1_bias_fixed_in, gnn_convs_1_mlp_linear_1_bias_fixed);
    copy_2d<64, 128>(gnn_convs_2_mlp_linear_0_weight_fixed_in, gnn_convs_2_mlp_linear_0_weight_fixed);
    copy_1d<64>(gnn_convs_2_mlp_linear_0_bias_fixed_in, gnn_convs_2_mlp_linear_0_bias_fixed);
    copy_2d<128, 64>(gnn_convs_2_mlp_linear_1_weight_fixed_in, gnn_convs_2_mlp_linear_1_weight_fixed);
    copy_1d<128>(gnn_convs_2_mlp_linear_1_bias_fixed_in, gnn_convs_2_mlp_linear_1_bias_fixed);
    copy_2d<64, 128>(gnn_convs_3_mlp_linear_0_weight_fixed_in, gnn_convs_3_mlp_linear_0_weight_fixed);
    copy_1d<64>(gnn_convs_3_mlp_linear_0_bias_fixed_in, gnn_convs_3_mlp_linear_0_bias_fixed);
    copy_2d<128, 64>(gnn_convs_3_mlp_linear_1_weight_fixed_in, gnn_convs_3_mlp_linear_1_weight_fixed);
    copy_1d<128>(gnn_convs_3_mlp_linear_1_bias_fixed_in, gnn_convs_3_mlp_linear_1_bias_fixed);
    copy_2d<64, 128>(gnn_convs_4_mlp_linear_0_weight_fixed_in, gnn_convs_4_mlp_linear_0_weight_fixed);
    copy_1d<64>(gnn_convs_4_mlp_linear_0_bias_fixed_in, gnn_convs_4_mlp_linear_0_bias_fixed);
    copy_2d<128, 64>(gnn_convs_4_mlp_linear_1_weight_fixed_in, gnn_convs_4_mlp_linear_1_weight_fixed);
    copy_1d<128>(gnn_convs_4_mlp_linear_1_bias_fixed_in, gnn_convs_4_mlp_linear_1_bias_fixed);
    copy_2d<64, 128>(gnn_convs_5_mlp_linear_0_weight_fixed_in, gnn_convs_5_mlp_linear_0_weight_fixed);
    copy_1d<64>(gnn_convs_5_mlp_linear_0_bias_fixed_in, gnn_convs_5_mlp_linear_0_bias_fixed);
    copy_2d<64, 64>(gnn_convs_5_mlp_linear_1_weight_fixed_in, gnn_convs_5_mlp_linear_1_weight_fixed);
    copy_1d<64>(gnn_convs_5_mlp_linear_1_bias_fixed_in, gnn_convs_5_mlp_linear_1_bias_fixed);
}

// Load Graph Data Function //
void load_graph_data(
    F_TYPE node_feature_table_in[600][11],
    int edge_list_in[600][2]
){
    #pragma HLS INLINE off
    
    copy_2d<600, 11>(node_feature_table_in, node_feature_table);
    copy_2d<600, 2>(edge_list_in, edge_list);
}

// Top Kernel //
void fpga_gin_qm9_top(
    F_TYPE node_feature_table_input[600][11],
    int edge_list_input[600][2],
    F_TYPE model_output[19],
    int num_of_nodes,
    int num_of_edges,
    int copy_parameters_flag,
    W_TYPE mlp_head_linear_layers_0_weight_fixed_in[64][192],
    W_TYPE mlp_head_linear_layers_0_bias_fixed_in[64],
    W_TYPE mlp_head_linear_layers_1_weight_fixed_in[64][64],
    W_TYPE mlp_head_linear_layers_1_bias_fixed_in[64],
    W_TYPE mlp_head_linear_layers_2_weight_fixed_in[64][64],
    W_TYPE mlp_head_linear_layers_2_bias_fixed_in[64],
    W_TYPE mlp_head_linear_layers_3_weight_fixed_in[64][64],
    W_TYPE mlp_head_linear_layers_3_bias_fixed_in[64],
    W_TYPE mlp_head_linear_layers_4_weight_fixed_in[19][64],
    W_TYPE mlp_head_linear_layers_4_bias_fixed_in[19],
    W_TYPE gnn_convs_0_mlp_linear_0_weight_fixed_in[64][11],
    W_TYPE gnn_convs_0_mlp_linear_0_bias_fixed_in[64],
    W_TYPE gnn_convs_0_mlp_linear_1_weight_fixed_in[128][64],
    W_TYPE gnn_convs_0_mlp_linear_1_bias_fixed_in[128],
    W_TYPE gnn_convs_1_mlp_linear_0_weight_fixed_in[64][128],
    W_TYPE gnn_convs_1_mlp_linear_0_bias_fixed_in[64],
    W_TYPE gnn_convs_1_mlp_linear_1_weight_fixed_in[128][64],
    W_TYPE gnn_convs_1_mlp_linear_1_bias_fixed_in[128],
    W_TYPE gnn_convs_2_mlp_linear_0_weight_fixed_in[64][128],
    W_TYPE gnn_convs_2_mlp_linear_0_bias_fixed_in[64],
    W_TYPE gnn_convs_2_mlp_linear_1_weight_fixed_in[128][64],
    W_TYPE gnn_convs_2_mlp_linear_1_bias_fixed_in[128],
    W_TYPE gnn_convs_3_mlp_linear_0_weight_fixed_in[64][128],
    W_TYPE gnn_convs_3_mlp_linear_0_bias_fixed_in[64],
    W_TYPE gnn_convs_3_mlp_linear_1_weight_fixed_in[128][64],
    W_TYPE gnn_convs_3_mlp_linear_1_bias_fixed_in[128],
    W_TYPE gnn_convs_4_mlp_linear_0_weight_fixed_in[64][128],
    W_TYPE gnn_convs_4_mlp_linear_0_bias_fixed_in[64],
    W_TYPE gnn_convs_4_mlp_linear_1_weight_fixed_in[128][64],
    W_TYPE gnn_convs_4_mlp_linear_1_bias_fixed_in[128],
    W_TYPE gnn_convs_5_mlp_linear_0_weight_fixed_in[64][128],
    W_TYPE gnn_convs_5_mlp_linear_0_bias_fixed_in[64],
    W_TYPE gnn_convs_5_mlp_linear_1_weight_fixed_in[64][64],
    W_TYPE gnn_convs_5_mlp_linear_1_bias_fixed_in[64]
){

    #pragma HLS INTERFACE m_axi depth=6600 port=node_feature_table_input offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=1200 port=edge_list_input offset=slave bundle=mem

    #pragma HLS INTERFACE m_axi depth=12288 port=mlp_head_linear_layers_0_weight_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=64 port=mlp_head_linear_layers_0_bias_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=4096 port=mlp_head_linear_layers_1_weight_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=64 port=mlp_head_linear_layers_1_bias_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=4096 port=mlp_head_linear_layers_2_weight_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=64 port=mlp_head_linear_layers_2_bias_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=4096 port=mlp_head_linear_layers_3_weight_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=64 port=mlp_head_linear_layers_3_bias_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=1216 port=mlp_head_linear_layers_4_weight_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=19 port=mlp_head_linear_layers_4_bias_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=704 port=gnn_convs_0_mlp_linear_0_weight_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=64 port=gnn_convs_0_mlp_linear_0_bias_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=8192 port=gnn_convs_0_mlp_linear_1_weight_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=128 port=gnn_convs_0_mlp_linear_1_bias_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=8192 port=gnn_convs_1_mlp_linear_0_weight_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=64 port=gnn_convs_1_mlp_linear_0_bias_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=8192 port=gnn_convs_1_mlp_linear_1_weight_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=128 port=gnn_convs_1_mlp_linear_1_bias_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=8192 port=gnn_convs_2_mlp_linear_0_weight_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=64 port=gnn_convs_2_mlp_linear_0_bias_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=8192 port=gnn_convs_2_mlp_linear_1_weight_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=128 port=gnn_convs_2_mlp_linear_1_bias_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=8192 port=gnn_convs_3_mlp_linear_0_weight_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=64 port=gnn_convs_3_mlp_linear_0_bias_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=8192 port=gnn_convs_3_mlp_linear_1_weight_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=128 port=gnn_convs_3_mlp_linear_1_bias_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=8192 port=gnn_convs_4_mlp_linear_0_weight_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=64 port=gnn_convs_4_mlp_linear_0_bias_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=8192 port=gnn_convs_4_mlp_linear_1_weight_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=128 port=gnn_convs_4_mlp_linear_1_bias_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=8192 port=gnn_convs_5_mlp_linear_0_weight_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=64 port=gnn_convs_5_mlp_linear_0_bias_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=4096 port=gnn_convs_5_mlp_linear_1_weight_fixed_in offset=slave bundle=mem
    #pragma HLS INTERFACE m_axi depth=64 port=gnn_convs_5_mlp_linear_1_bias_fixed_in offset=slave bundle=mem


    #pragma HLS bind_storage variable=node_feature_table type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=edge_list type=RAM_2P impl=bram

    #pragma HLS bind_storage variable=mlp_head_linear_layers_0_weight_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=mlp_head_linear_layers_0_bias_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=mlp_head_linear_layers_1_weight_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=mlp_head_linear_layers_1_bias_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=mlp_head_linear_layers_2_weight_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=mlp_head_linear_layers_2_bias_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=mlp_head_linear_layers_3_weight_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=mlp_head_linear_layers_3_bias_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=mlp_head_linear_layers_4_weight_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=mlp_head_linear_layers_4_bias_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=gnn_convs_0_mlp_linear_0_weight_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=gnn_convs_0_mlp_linear_0_bias_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=gnn_convs_0_mlp_linear_1_weight_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=gnn_convs_0_mlp_linear_1_bias_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=gnn_convs_1_mlp_linear_0_weight_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=gnn_convs_1_mlp_linear_0_bias_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=gnn_convs_1_mlp_linear_1_weight_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=gnn_convs_1_mlp_linear_1_bias_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=gnn_convs_2_mlp_linear_0_weight_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=gnn_convs_2_mlp_linear_0_bias_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=gnn_convs_2_mlp_linear_1_weight_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=gnn_convs_2_mlp_linear_1_bias_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=gnn_convs_3_mlp_linear_0_weight_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=gnn_convs_3_mlp_linear_0_bias_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=gnn_convs_3_mlp_linear_1_weight_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=gnn_convs_3_mlp_linear_1_bias_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=gnn_convs_4_mlp_linear_0_weight_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=gnn_convs_4_mlp_linear_0_bias_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=gnn_convs_4_mlp_linear_1_weight_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=gnn_convs_4_mlp_linear_1_bias_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=gnn_convs_5_mlp_linear_0_weight_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=gnn_convs_5_mlp_linear_0_bias_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=gnn_convs_5_mlp_linear_1_weight_fixed type=RAM_2P impl=bram
    #pragma HLS bind_storage variable=gnn_convs_5_mlp_linear_1_bias_fixed type=RAM_2P impl=bram


    int num_of_nodes_top = num_of_nodes;
    int num_of_edges_top = num_of_edges;
    int copy_parameters_flag_top = copy_parameters_flag;
    
    if (copy_parameters_flag_top) {
        load_parameters(
            mlp_head_linear_layers_0_weight_fixed_in,
            mlp_head_linear_layers_0_bias_fixed_in,
            mlp_head_linear_layers_1_weight_fixed_in,
            mlp_head_linear_layers_1_bias_fixed_in,
            mlp_head_linear_layers_2_weight_fixed_in,
            mlp_head_linear_layers_2_bias_fixed_in,
            mlp_head_linear_layers_3_weight_fixed_in,
            mlp_head_linear_layers_3_bias_fixed_in,
            mlp_head_linear_layers_4_weight_fixed_in,
            mlp_head_linear_layers_4_bias_fixed_in,
            gnn_convs_0_mlp_linear_0_weight_fixed_in,
            gnn_convs_0_mlp_linear_0_bias_fixed_in,
            gnn_convs_0_mlp_linear_1_weight_fixed_in,
            gnn_convs_0_mlp_linear_1_bias_fixed_in,
            gnn_convs_1_mlp_linear_0_weight_fixed_in,
            gnn_convs_1_mlp_linear_0_bias_fixed_in,
            gnn_convs_1_mlp_linear_1_weight_fixed_in,
            gnn_convs_1_mlp_linear_1_bias_fixed_in,
            gnn_convs_2_mlp_linear_0_weight_fixed_in,
            gnn_convs_2_mlp_linear_0_bias_fixed_in,
            gnn_convs_2_mlp_linear_1_weight_fixed_in,
            gnn_convs_2_mlp_linear_1_bias_fixed_in,
            gnn_convs_3_mlp_linear_0_weight_fixed_in,
            gnn_convs_3_mlp_linear_0_bias_fixed_in,
            gnn_convs_3_mlp_linear_1_weight_fixed_in,
            gnn_convs_3_mlp_linear_1_bias_fixed_in,
            gnn_convs_4_mlp_linear_0_weight_fixed_in,
            gnn_convs_4_mlp_linear_0_bias_fixed_in,
            gnn_convs_4_mlp_linear_1_weight_fixed_in,
            gnn_convs_4_mlp_linear_1_bias_fixed_in,
            gnn_convs_5_mlp_linear_0_weight_fixed_in,
            gnn_convs_5_mlp_linear_0_bias_fixed_in,
            gnn_convs_5_mlp_linear_1_weight_fixed_in,
            gnn_convs_5_mlp_linear_1_bias_fixed_in
        );
    }

    load_graph_data(
        node_feature_table_input,
        edge_list_input
    );

    compute_degree_tables<600, 600, 12, 24>(
        edge_list,
        in_degree_table,
        out_degree_table,
        num_of_nodes_top,
        num_of_edges_top
    );

    compute_neighbor_tables<
        600,
        600,
        12,
        24
    >(
        edge_list,
        in_degree_table,
        out_degree_table,
        neightbor_table_offsets,
        neighbor_table,
        num_of_nodes_top,
        num_of_edges_top
    );

    compute_gnn_head(num_of_nodes_top, num_of_edges_top);
    compute_global_graph_pooling(num_of_nodes_top, num_of_edges_top);
    compute_mlp_head();
    compute_model_output();

    copy_1d<19, F_TYPE>(model_output_buffer, model_output);
}