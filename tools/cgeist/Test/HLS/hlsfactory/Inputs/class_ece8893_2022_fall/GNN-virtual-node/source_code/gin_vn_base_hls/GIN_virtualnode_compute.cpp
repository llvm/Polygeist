#include "dcl.hpp"

// #define _PRINT_

/// graph information
int node_feature[MAX_NODE * ND_FEATURE];
int edge_attr[MAX_EDGE][EDGE_ATTR];
int edge_list[MAX_EDGE * 2];

/// MLP data and message buffer
FM_TYPE message[MAX_NODE][EMB_DIM];
FM_TYPE mlp_in[MAX_NODE][EMB_DIM];
//FM_TYPE mlp_out[MAX_NODE][EMB_DIM];

// Virtual node mlp buffer
FM_TYPE virtualnode_mlp_in[EMB_DIM];

/// graph data
FM_TYPE graph_embedding[EMB_DIM];

/// intermediate node and edge embedding buffer
//FM_TYPE edge_embedding[MAX_EDGE][EMB_DIM];
FM_TYPE node_embedding[MAX_NODE][EMB_DIM];
// Barfi: modified
FM_TYPE virtualnode_embedding_weight[MLP_2_OUT]; //Copy of embedding weight in BRAM (done only once)
FM_TYPE virtualnode_embedding[MLP_2_OUT]; // Updated VN embedding every layer

/// embedding tables
WT_TYPE node_embedding_table[ND_FEATURE_TOTAL][EMB_DIM];
WT_TYPE edge_embedding_table[EG_FEATURE_TOTAL][EMB_DIM];

/// MLP related weights
WT_TYPE mlp_eps[LAYER_NUM];
WT_TYPE mlp_1_weights[LAYER_NUM][MLP_1_OUT][MLP_1_IN];
WT_TYPE mlp_1_bias[LAYER_NUM][MLP_1_OUT];
WT_TYPE mlp_2_weights[LAYER_NUM][MLP_2_OUT][MLP_2_IN];
WT_TYPE mlp_2_bias[LAYER_NUM][MLP_2_OUT];

/// Virtual Node MLP related weights
WT_TYPE virtualnode_mlp_1_weights[VN_LAYER_NUM][VN_MLP_1_OUT][VN_MLP_1_IN];
WT_TYPE virtualnode_mlp_1_bias[VN_LAYER_NUM][VN_MLP_1_OUT];
WT_TYPE virtualnode_mlp_2_weights[VN_LAYER_NUM][VN_MLP_2_OUT][VN_MLP_2_IN];
WT_TYPE virtualnode_mlp_2_bias[VN_LAYER_NUM][VN_MLP_2_OUT];

/// graph pred linear weights
WT_TYPE graph_pred_weights[NUM_TASK][MLP_2_OUT];
WT_TYPE graph_pred_bias[NUM_TASK];

int nd_feature_table[ND_FEATURE] = {119, 4, 12, 12, 10, 6, 6, 2, 2};
int ed_feature_table[EDGE_ATTR] = {5, 6, 2};


void MLP_one_node_one_dim(int dim, int nd, FM_TYPE mlp_in[MAX_NODE][EMB_DIM], FM_TYPE mlp_out[MAX_NODE][EMB_DIM], int layer)
{
    FM_TYPE psum[MLP_1_IN];

#pragma HLS array_partition variable=psum complete
#pragma HLS array_partition variable=mlp_in dim=2 complete
#pragma HLS array_partition variable=mlp_out dim=2 complete
#pragma HLS array_partition variable=mlp_1_weights dim=3 complete
// #pragma HLS array_partition variable=mlp_1_bias dim=2 complete
#pragma HLS array_partition variable=mlp_2_weights dim=2 complete
// #pragma HLS array_partition variable=mlp_2_bias dim=2 complete


    // first layer of 300 x 300 VVM
    FM_TYPE sum = mlp_1_bias[layer][dim];
    for(int dim_in1 = 0; dim_in1 < MLP_1_IN; dim_in1++) {
        psum[dim_in1] = mlp_1_weights[layer][dim][dim_in1] * mlp_in[nd][dim_in1];
        sum += psum[dim_in1];
    }
    sum = sum < 0 ? (FM_TYPE)0 : sum;

    // second layer of 300 x 300 VVM
    for(int dim_out2 = 0; dim_out2 < MLP_2_OUT; dim_out2++) {
        mlp_out[nd][dim_out2] += sum * mlp_2_weights[layer][dim_out2][dim];
    }

}

void MLP(FM_TYPE mlp_in[MAX_NODE][EMB_DIM],FM_TYPE virtualnode_mlp_in[EMB_DIM],  FM_TYPE node_embedding[MAX_NODE][EMB_DIM], FM_TYPE h[MAX_NODE][EMB_DIM], FM_TYPE virtualnode_embedding[EMB_DIM], int num_of_nodes, int layer)
{
#pragma HLS inline off
#pragma HLS array_partition variable=node_embedding complete dim=2
#pragma HLS array_partition variable=virtualnode_embedding complete
    /// something special in GIN
    WT_TYPE _eps = mlp_eps[layer];

    /// MLP input by aggregating messages and self features
    for(int dim = 0; dim < EMB_DIM; dim++) {
        for(int nd = 0; nd < num_of_nodes; nd++) {
            mlp_in[nd][dim] = message[nd][dim] + (1 + _eps) * h[nd][dim];
        }
    }
    
    // Separate loop for virtual node mlp initialization to see if this is causing an issue
    for(int dim = 0; dim < EMB_DIM; dim++) {
        virtualnode_mlp_in[dim] = virtualnode_embedding[dim];
        for(int nd = 0; nd < num_of_nodes; nd++) {
            // Form Virtual node MLP input by aggregating all nodes
            virtualnode_mlp_in[dim] += h[nd][dim];
        }
    }

#ifdef _PRINT_
    printf("\nInput of MLP\n");
    for(int nd = 0; nd < 5; nd++) {
        printf("Node %d: ", nd);
        for(int dim = 0; dim < 10; dim++) {
            printf("%.5f ", mlp_in[nd][dim].to_float());
        }
        printf("...\n");
    }
#endif 


    //memset(node_embedding, 0, num_of_nodes * EMB_DIM * sizeof(FM_TYPE));
    for(int nd = 0; nd < num_of_nodes; nd++) {
        for(int dim = 0; dim < EMB_DIM; dim++) {
            node_embedding[nd][dim] = 0;
        }
    }

    for(int dim = 0; dim < MLP_1_OUT; dim++) {
        for(int nd = 0; nd < num_of_nodes; nd++) {
#pragma HLS pipeline
            MLP_one_node_one_dim(dim, nd, mlp_in, node_embedding, layer);
        }
    }

    for(int nd = 0; nd < num_of_nodes; nd++) {
#pragma HLS pipeline
        for(int dim = 0; dim < EMB_DIM; dim++) {
            if( node_embedding[nd][dim] + mlp_2_bias[layer][dim] < 0 && layer != 4 ) {
                node_embedding[nd][dim] = 0;
            }
            else {
                node_embedding[nd][dim] = node_embedding[nd][dim] + mlp_2_bias[layer][dim];
            }
        }
    }



#ifdef _PRINT_
    printf("\nOutput of MLP\n");
    for(int nd = 0; nd < 5; nd++) {
        printf("Node %d: ", nd);
        for(int dim = 0; dim < 10; dim++) {
            printf("%.5f ", node_embedding[nd][dim].to_float());
        }
        printf("...\n");
    }
#endif
}

void virtualnode_MLP(FM_TYPE virtualnode_mlp_in[EMB_DIM], FM_TYPE virtualnode_embedding[EMB_DIM], int layer)
{
#pragma HLS inline off
#pragma HLS array_partition variable=virtualnode_mlp_in complete
#pragma HLS array_partition variable=virtualnode_mlp_1_weights dim=3 complete
#pragma HLS array_partition variable=virtualnode_mlp_2_weights dim=2 complete
#pragma HLS array_partition variable=virtualnode_embedding complete
//#pragma HLS array_partition variable=virtualnode_mlp_2_bias complete dim=2

    for(int dim = 0; dim < EMB_DIM; dim++) {
    #pragma HLS unroll
       virtualnode_embedding[dim] = virtualnode_mlp_2_bias[layer][dim];
    }
    

#ifdef _PRINT_
    printf("\nInput of virtual node MLP\n");
    for(int dim = 0; dim < 10; dim++) {
        printf("%.5f ", virtualnode_mlp_in[dim].to_float());
    }
#endif 

    for(int dim = 0; dim < VN_MLP_1_OUT; dim++) {
        FM_TYPE psum[VN_MLP_1_IN];

#pragma HLS array_partition variable=psum complete
#pragma HLS pipeline

        FM_TYPE sum = virtualnode_mlp_1_bias[layer][dim];
        
        // Compute one dimension of hidden MLP layer for virtual node
        for(int dim_in1 = 0; dim_in1 < VN_MLP_1_IN; dim_in1++) {
            psum[dim_in1] = virtualnode_mlp_1_weights[layer][dim][dim_in1] * virtualnode_mlp_in[dim_in1];
            sum += psum[dim_in1];
        }
        sum = sum < 0 ? (FM_TYPE)0 : sum;
        
        //Broadcast the output of that entry to output MLP layer of virtual node
        for(int dim_out2 = 0; dim_out2 < VN_MLP_2_OUT; dim_out2++) {
            virtualnode_embedding[dim_out2] += sum * virtualnode_mlp_2_weights[layer][dim_out2][dim];
        }
    }

    for(int dim = 0; dim < EMB_DIM; dim++) {
        #pragma HLS unroll
        if(virtualnode_embedding[dim] < (FM_TYPE)0) {
           virtualnode_embedding[dim] = (FM_TYPE)0;
        }
    }

#ifdef _PRINT_
    printf("\nOutput of virtualnode MLP\n");
    for(int dim = 0; dim < 10; dim++) {
        printf("%.5f ", virtualnode_embedding[dim].to_float());
    }
#endif
}

int get_ed_emb_addr(int ef, int layer)
{
    int addr = 0;
    for(int i = 0; i < ef; i++) {
        addr += ed_feature_table[i];
    }
    return addr + layer * (5+6+2);
}



void compute_edge_embedding_and_message_passing(int num_of_nodes, int num_of_edges, int edge_attr[MAX_EDGE][EDGE_ATTR], int layer)
{
#pragma HLS inline off

//#pragma HLS array_partition variable=message dim=2 complete
#pragma HLS array_partition variable=edge_embedding_table complete
#pragma HLS array_partition variable=edge_attr dim=2 complete
#pragma HLS array_partition variable=node_embedding dim=2 complete
#pragma HLS array_partition variable=virtualnode_embedding complete

    ////////////// Embedding: compute edge embedding
    memset(message, 0, num_of_nodes * EMB_DIM * sizeof(FM_TYPE));
    // Initialize message to virtual node embedding for the given layer
    for(int nd = 0; nd < num_of_nodes; nd++)
    {
    #pragma HLS pipeline
        for(int dim = 0; dim < EMB_DIM; dim++)
        {
            node_embedding[nd][dim] += virtualnode_embedding[dim];
        }
    }

    for(int e = 0; e < num_of_edges; e++) {
        int u = edge_list[e*2];     // source node id
        int v = edge_list[e*2+1];   // target node id

        for(int dim = 0; dim < EMB_DIM; dim++) {
#pragma HLS pipeline
            FM_TYPE edge_embed = 0;

            for(int ef = 0; ef < EDGE_ATTR; ef++) {
                int e_f = edge_attr[e][ef];
                int addr = get_ed_emb_addr(ef, layer);
                FM_TYPE emb_value = 0;
                emb_value = edge_embedding_table[addr + e_f][dim];
                edge_embed += emb_value;

            }   
            FM_TYPE msg = edge_embed + node_embedding[u][dim];
            if(msg < 0) msg = 0.0;
            message[v][dim] += msg;   
        }
    }
}

int get_nd_emb_addr(int nf)
{
    int addr = 0;
    for(int i = 0; i < nf; i++) {
        addr += nd_feature_table[i];
    }
    return addr;
}


void compute_node_embedding(int num_of_nodes, int* features)
{
#pragma HLS inline off
    ////////////// Embedding: compute input node embedding
//    memset(node_embedding, 0, num_of_nodes * EMB_DIM * sizeof(FM_TYPE));

    for(int nd = 0; nd < num_of_nodes; nd++) {
        for(int dim = 0; dim < EMB_DIM; dim++) {
        node_embedding[nd][dim] = 0;
    }
    }


    for(int nd = 0; nd < num_of_nodes; nd++) {
        for(int nf = 0; nf < ND_FEATURE; nf++) {
            int nd_f = features[nd * ND_FEATURE + nf];
            int emb_addr = get_nd_emb_addr(nf);

            for(int dim = 0; dim < EMB_DIM; dim += 2) {
// TODO: See if unrolling is required?
#pragma HLS pipeline II=2
                FM_TYPE emb_value1, emb_value2;    
                emb_value1 = node_embedding_table[emb_addr + nd_f][dim]; 
                node_embedding[nd][dim] += emb_value1;
                emb_value2 = node_embedding_table[emb_addr + nd_f][dim + 1]; 
                node_embedding[nd][dim + 1] += emb_value2;
            }   
        }
    }

#ifdef _PRINT_
    printf("\nInitial node embedding:\n");
    for(int nd = 0; nd < 5; nd++) {
        printf("Node %d: ", nd);
        for(int dim = 0; dim < 10; dim++) {
            printf("%.5f ", node_embedding[nd][dim].to_float());
        }
        printf("...\n");
    }
#endif
}

// Loading virtual node embedding weights from DRAM to BRAM. Done only once
void load_virtualnode_embedding(FM_TYPE* gnn_node_virtualnode_embedding_weight_fixed)
{
#pragma HLS inline off
    for(int dim = 0; dim < EMB_DIM; dim++) 
    {
        virtualnode_embedding_weight[dim] = gnn_node_virtualnode_embedding_weight_fixed[dim];
    }
}

// Initialize virtual node embedding
void initialize_virtualnode_embedding()
{
#pragma HLS inline off
#pragma HLS array_partition variable=virtualnode_embedding complete 
#pragma HLS array_partition variable=virtualnode_embedding_weight complete 
    for(int dim = 0; dim < EMB_DIM; dim++) 
    {
        virtualnode_embedding[dim] = virtualnode_embedding_weight[dim];
    }
#ifdef _PRINT_
    printf("\nInitial virtual node embedding:\n");
    for(int dim = 0; dim < 10; dim++) {
        printf("%.5f ", virtualnode_embedding[dim].to_float());
    }
#endif
}

void load_mlp_weights_one_layer(int layer, FM_TYPE* gnn_node_mlp_1_weights_fixed, FM_TYPE* gnn_node_mlp_1_bias_fixed,
                                           FM_TYPE* gnn_node_mlp_2_weights_fixed, FM_TYPE* gnn_node_mlp_2_bias_fixed)
{
#pragma HLS inline off

    for(int dim_out = 0; dim_out < MLP_1_OUT; dim_out++) {
        mlp_1_bias[layer][dim_out] = gnn_node_mlp_1_bias_fixed[layer * MLP_1_OUT + dim_out];
        for(int dim_in = 0; dim_in < MLP_1_IN; dim_in++) {
            mlp_1_weights[layer][dim_out][dim_in] = gnn_node_mlp_1_weights_fixed[layer * MLP_1_OUT * MLP_1_IN + dim_out * MLP_1_IN + dim_in];
        }
    }        

    for(int dim_out = 0; dim_out < MLP_2_OUT; dim_out++) {
        mlp_2_bias[layer][dim_out] = gnn_node_mlp_2_bias_fixed[layer * MLP_2_OUT + dim_out];
        for(int dim_in = 0; dim_in < MLP_2_IN; dim_in++) {
            mlp_2_weights[layer][dim_out][dim_in] = gnn_node_mlp_2_weights_fixed[layer * MLP_2_OUT * MLP_2_IN + dim_out * MLP_2_IN + dim_in];
        }
    }
}

// Loading virtual node mlp weights from DRAM to BRAM. Done only once
void load_virtualnode_mlp_weights_one_layer(int layer, FM_TYPE* gnn_node_virtualnode_mlp_1_weights_fixed, FM_TYPE* gnn_node_virtualnode_mlp_1_bias_fixed,
                                           FM_TYPE* gnn_node_virtualnode_mlp_2_weights_fixed, FM_TYPE* gnn_node_virtualnode_mlp_2_bias_fixed)
{
#pragma HLS inline off

    for(int dim_out = 0; dim_out < VN_MLP_1_OUT; dim_out++) {
        virtualnode_mlp_1_bias[layer][dim_out] = gnn_node_virtualnode_mlp_1_bias_fixed[layer * VN_MLP_1_OUT + dim_out];
        for(int dim_in = 0; dim_in < VN_MLP_1_IN; dim_in++) {
            virtualnode_mlp_1_weights[layer][dim_out][dim_in] = gnn_node_virtualnode_mlp_1_weights_fixed[layer * VN_MLP_1_OUT * VN_MLP_1_IN + dim_out * VN_MLP_1_IN + dim_in];
        }
    }

    for(int dim_out = 0; dim_out < VN_MLP_2_OUT; dim_out++) {
        virtualnode_mlp_2_bias[layer][dim_out] = gnn_node_virtualnode_mlp_2_bias_fixed[layer * VN_MLP_2_OUT + dim_out];
        for(int dim_in = 0; dim_in < VN_MLP_2_IN; dim_in++) {
            virtualnode_mlp_2_weights[layer][dim_out][dim_in] = gnn_node_virtualnode_mlp_2_weights_fixed[layer * VN_MLP_2_OUT * VN_MLP_2_IN + dim_out * VN_MLP_2_IN + dim_in];
        }
    }        
}

void compute_CONV_layer(FM_TYPE h_node[MAX_NODE][EMB_DIM], int num_of_nodes, int num_of_edges, int layer)
{
#pragma HLS inline off
    
    compute_edge_embedding_and_message_passing(num_of_nodes, num_of_edges, edge_attr, layer);

    MLP(mlp_in, virtualnode_mlp_in, node_embedding, h_node, virtualnode_embedding, num_of_nodes, layer);

#ifdef _PRINT_
    printf("\nOutput of Conv %d\n", layer);
    for(int nd = 0; nd < 5; nd++) {
        printf("Node %d: ", nd);
        for(int dim = 0; dim < 10; dim++) {
            printf("%.5f ", h_node[nd][dim].to_float());
        }
        printf("...\n");
    }
#endif
}

void global_mean_pooling(FM_TYPE* h_graph, FM_TYPE h_node[MAX_NODE][EMB_DIM], int num_of_nodes)
{
#pragma HLS inline off

    FM_TYPE h_node_sum[EMB_DIM];
    memset(h_node_sum, 0, EMB_DIM * sizeof(FM_TYPE));

    
    for(int dim = 0; dim < EMB_DIM; dim++) {
        WT_TYPE sum = 0;
        for(int nd = 0; nd < num_of_nodes; nd++) {
            sum += h_node[nd][dim];
        }
        h_graph[dim] = sum / num_of_nodes;
    }

#ifdef _PRINT_
    printf("\nGlobal h_graph (global mean pool):\n");
    for(int dim = 0; dim < 10; dim++) {
        printf("%.5f ", h_graph[dim].to_float());
    }
    printf("...\n");
#endif
}


void global_graph_prediction(FM_TYPE* d_out, FM_TYPE* d_in)
{
#pragma HLS inline off
    for(int tsk = 0; tsk < NUM_TASK; tsk++) {
        d_out[tsk] = graph_pred_bias[tsk];
        for(int dim = 0; dim < EMB_DIM; dim++) {
            d_out[tsk] += d_in[dim] * graph_pred_weights[tsk][dim];
        }
    }
}


void load_graph(int* node_feature, int edge_attr[MAX_EDGE][EDGE_ATTR], int* edge_list, int* node_feature_in, int* edge_list_in, int* edge_attr_in, int num_of_nodes, int num_of_edges)
{
#pragma HLS inline off
    for(int i = 0; i < num_of_nodes * ND_FEATURE; i++) {
        node_feature[i] = node_feature_in[i];
    }
    
    for(int e = 0; e < num_of_edges; e++) {
        for(int i = 0; i < EDGE_ATTR; i++) {
            edge_attr[e][i] = edge_attr_in[e * EDGE_ATTR + i];
        }
    }

    for(int i = 0; i < num_of_edges * 2; i++) {
        edge_list[i] = edge_list_in[i];
    }
}


/// these weights will be loaded once and stay in BRAM forever
void load_misc_weights(
    WT_TYPE eps_in[LAYER_NUM],
    WT_TYPE graph_pred_weight_in[NUM_TASK * MLP_2_OUT],
    WT_TYPE graph_pred_bias_in[NUM_TASK],
    WT_TYPE node_embedding_table_in[ND_FEATURE_TOTAL * EMB_DIM],
    WT_TYPE edge_embedding_table_in[EG_FEATURE_TOTAL * EMB_DIM])
{
#pragma HLS inline off
    for(int i = 0; i < LAYER_NUM; i++) {
        mlp_eps[i] = eps_in[i];
    }
    
    for(int t = 0; t < NUM_TASK; t++) {
        graph_pred_bias[t] = graph_pred_bias_in[t];
        for(int dim_in = 0; dim_in < MLP_2_OUT; dim_in++ ) {
            graph_pred_weights[t][dim_in] = graph_pred_weight_in[t * MLP_2_OUT + dim_in];
        }
    }

    for(int i = 0; i < ND_FEATURE_TOTAL; i++) {
        for(int dim = 0; dim < EMB_DIM; dim++) {    
            node_embedding_table[i][dim] = node_embedding_table_in[i * EMB_DIM + dim];
        }
    }

    for(int i = 0; i < EG_FEATURE_TOTAL; i++) {
        for(int dim = 0; dim < EMB_DIM; dim++) {
            edge_embedding_table[i][dim] = edge_embedding_table_in[i * EMB_DIM + dim];
        }
    }
}


extern "C" {
void GIN_virtualnode_compute_one_graph(
    int* node_feature_in, int* edge_list_in, int* edge_attr_in, int* graph_attr, FM_TYPE* task,
    WT_TYPE gnn_node_mlp_1_weights_fixed[LAYER_NUM * MLP_1_OUT * MLP_1_IN],
    WT_TYPE gnn_node_mlp_1_bias_fixed[LAYER_NUM * MLP_1_OUT],
    WT_TYPE gnn_node_mlp_2_weights_fixed[LAYER_NUM * MLP_2_OUT * MLP_2_IN],
    WT_TYPE gnn_node_mlp_2_bias_fixed[LAYER_NUM * MLP_2_OUT],
    WT_TYPE gnn_node_embedding_fixed[ND_FEATURE_TOTAL * EMB_DIM],
    WT_TYPE gnn_edge_embedding_fixed[EG_FEATURE_TOTAL * EMB_DIM],
    WT_TYPE graph_pred_linear_weight_fixed[NUM_TASK * MLP_2_OUT],
    WT_TYPE graph_pred_linear_bias_fixed[NUM_TASK],
    WT_TYPE eps_fixed[LAYER_NUM],
    FM_TYPE gnn_node_virtualnode_embedding_weight_fixed[1 * EMB_DIM],
    WT_TYPE gnn_node_virtualnode_mlp_1_weights_fixed[VN_LAYER_NUM * VN_MLP_1_OUT * VN_MLP_1_IN],
    WT_TYPE gnn_node_virtualnode_mlp_1_bias_fixed[VN_LAYER_NUM * VN_MLP_1_OUT],
    WT_TYPE gnn_node_virtualnode_mlp_2_weights_fixed[VN_LAYER_NUM * VN_MLP_2_OUT * VN_MLP_2_IN], 
    WT_TYPE gnn_node_virtualnode_mlp_2_bias_fixed[VN_LAYER_NUM * VN_MLP_2_OUT])
{
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS INTERFACE m_axi depth=100000 port=node_feature_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=100000 port=edge_list_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=100000 port=edge_attr_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=100000 port=graph_attr offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=100000 port=task offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=100000 port=gnn_node_mlp_1_weights_fixed offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=100000 port=gnn_node_mlp_1_bias_fixed offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=100000 port=gnn_node_mlp_2_weights_fixed offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=100000 port=gnn_node_mlp_2_bias_fixed offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=100000 port=gnn_node_embedding_fixed offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=100000 port=gnn_edge_embedding_fixed offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=100000 port=graph_pred_linear_weight_fixed offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=100000 port=graph_pred_linear_bias_fixed offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=100000 port=eps_fixed offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=100000 port=gnn_node_virtualnode_mlp_1_weights_fixed offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=100000 port=gnn_node_virtualnode_mlp_1_bias_fixed offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=100000 port=gnn_node_virtualnode_mlp_2_weights_fixed offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=100000 port=gnn_node_virtualnode_mlp_2_bias_fixed offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=100000 port=gnn_node_virtualnode_embedding_weight_fixed offset=slave bundle=mem

#pragma HLS bind_storage variable=node_feature type=RAM_2P impl=bram
#pragma HLS bind_storage variable=edge_attr type=RAM_2P impl=bram
#pragma HLS bind_storage variable=edge_list type=RAM_2P impl=bram
#pragma HLS bind_storage variable=mlp_in type=RAM_2P impl=bram
//#pragma HLS bind_storage variable=mlp_out type=RAM_2P impl=bram
#pragma HLS bind_storage variable=graph_embedding type=RAM_2P impl=bram
//#pragma HLS bind_storage variable=edge_embedding type=RAM_2P impl=uram
#pragma HLS bind_storage variable=node_embedding type=RAM_2P impl=bram
#pragma HLS bind_storage variable=node_embedding_table type=RAM_2P impl=bram
#pragma HLS bind_storage variable=edge_embedding_table type=RAM_2P impl=bram
#pragma HLS bind_storage variable=message type=RAM_2P impl=uram
// #pragma HLS bind_storage variable=virtualnode_mlp_in type=RAM_2P impl=bram
// #pragma HLS bind_storage variable=virtualnode_embedding type=RAM_2P impl=bram
#pragma HLS bind_storage variable=virtualnode_embedding_weight type=RAM_2P impl=bram

    int num_of_nodes = graph_attr[0];
    int num_of_edges = graph_attr[1];
    int is_first = graph_attr[2]; //is the first graph

    num_of_nodes = 19;
    num_of_edges = 40;
    // is_first = 0;

    if( is_first == 1 ) {
        ////////////// Load weights
        for(int layer = 0; layer < VN_LAYER_NUM; layer++) {
            load_mlp_weights_one_layer(layer, gnn_node_mlp_1_weights_fixed, gnn_node_mlp_1_bias_fixed, gnn_node_mlp_2_weights_fixed, gnn_node_mlp_2_bias_fixed);
            load_virtualnode_mlp_weights_one_layer(layer, gnn_node_virtualnode_mlp_1_weights_fixed, gnn_node_virtualnode_mlp_1_bias_fixed, 
                gnn_node_virtualnode_mlp_2_weights_fixed, gnn_node_virtualnode_mlp_2_bias_fixed);
        }

        // Load the last layer of mlp weights for regular nodes
        load_mlp_weights_one_layer(LAYER_NUM - 1, gnn_node_mlp_1_weights_fixed, gnn_node_mlp_1_bias_fixed, gnn_node_mlp_2_weights_fixed, gnn_node_mlp_2_bias_fixed);
        
        load_misc_weights(eps_fixed, graph_pred_linear_weight_fixed, graph_pred_linear_bias_fixed,
                          gnn_node_embedding_fixed, gnn_edge_embedding_fixed);

        load_virtualnode_embedding(gnn_node_virtualnode_embedding_weight_fixed);
    }


    ///////////// Load a new graph onto chip
    load_graph(node_feature, edge_attr, edge_list, node_feature_in, edge_list_in, edge_attr_in, num_of_nodes, num_of_edges);

    printf("Computing GIN Virtual Node...\n");

    ////////////// Embedding: compute input node embedding and initialize virtual node embedding
    compute_node_embedding(num_of_nodes, node_feature);
    
    /// Initialize virtual node embedding
    initialize_virtualnode_embedding();

    ////////////// CONV layers //////////////////////////////////
    for(int layer = 0; layer < VN_LAYER_NUM; layer++) {
        compute_CONV_layer(node_embedding, num_of_nodes, num_of_edges, layer);
        virtualnode_MLP(virtualnode_mlp_in, virtualnode_embedding, layer);
    }

    compute_CONV_layer(node_embedding, num_of_nodes, num_of_edges, LAYER_NUM - 1);
        
    ////////////// Global mean pooling //////////////////////
    global_mean_pooling(graph_embedding, node_embedding, num_of_nodes);
    
    ////////////// Graph prediction linear ///////////////////
    global_graph_prediction(task, graph_embedding);


    printf("Final graph prediction:\n");
    for(int tsk = 0; tsk < NUM_TASK; tsk++) {
        printf("%.7f\n", task[tsk].to_float());
    }
    printf("\nGIN Virtual Node computation done.\n");

}
}
