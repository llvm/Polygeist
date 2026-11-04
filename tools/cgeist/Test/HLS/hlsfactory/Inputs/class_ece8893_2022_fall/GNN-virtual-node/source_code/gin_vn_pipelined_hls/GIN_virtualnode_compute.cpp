#include "dcl.hpp"

// For debug purposes
// #define _PRINT_

/// graph information
int node_feature[MAX_NODE * ND_FEATURE];
int edge_attr[MAX_EDGE][EDGE_ATTR];
int edge_list[MAX_EDGE * 2];

// Neighbor table and degree table for regularising
// memory accesses in message passing
int degree_table[MAX_NODE * 3];
int neighbor_table[MAX_NODE * MAX_DEGREE * 2];

/// MLP data and message buffer
FM_TYPE message1[MAX_NODE][EMB_DIM]; // need two tables for storing the message
FM_TYPE message2[MAX_NODE][EMB_DIM]; // need two tables for storing the message

// graph data
FM_TYPE graph_embedding[EMB_DIM];

// intermediate node embedding buffer
FM_TYPE node_embedding[MAX_NODE][EMB_DIM];

/// embedding tables (atom and bond encoder weights)
WT_TYPE node_embedding_table[ND_FEATURE_TOTAL][EMB_DIM];
WT_TYPE edge_embedding_table[EG_FEATURE_TOTAL][EMB_DIM];

/// MLP related weights
WT_TYPE mlp_eps[LAYER_NUM];
WT_TYPE mlp_1_weights[LAYER_NUM][MLP_1_OUT][MLP_1_IN];
WT_TYPE mlp_1_bias[LAYER_NUM][MLP_1_OUT];
WT_TYPE mlp_2_weights[LAYER_NUM][MLP_2_OUT][MLP_2_IN];
WT_TYPE mlp_2_bias[LAYER_NUM][MLP_2_OUT];

// Virtual node embedding weight
FM_TYPE virtualnode_embedding_weight[EMB_DIM];

// Intermediate Virtual Node Embeddings
FM_TYPE vn_embedding1[EMB_DIM];
FM_TYPE vn_embedding2[EMB_DIM];

// Virtual Node MLP related weights
WT_TYPE virtualnode_mlp_1_weights[VN_LAYER_NUM][VN_MLP_1_OUT][VN_MLP_1_IN];
WT_TYPE virtualnode_mlp_1_bias[VN_LAYER_NUM][VN_MLP_1_OUT];
WT_TYPE virtualnode_mlp_2_weights[VN_LAYER_NUM][VN_MLP_2_OUT][VN_MLP_2_IN];
WT_TYPE virtualnode_mlp_2_bias[VN_LAYER_NUM][VN_MLP_2_OUT];

// graph pred linear weights
WT_TYPE graph_pred_weights[NUM_TASK][MLP_2_OUT];
WT_TYPE graph_pred_bias[NUM_TASK];

int nd_feature_table[ND_FEATURE] = {119, 4, 12, 12, 10, 6, 6, 2, 2};
int ed_feature_table[EDGE_ATTR] = {5, 6, 2};

void clear_message_table(FM_TYPE message_tb[MAX_NODE][EMB_DIM], int num_of_nodes);

void MLP_one_node(int nd, FM_TYPE mlp_in[EMB_DIM], FM_TYPE mlp_out[EMB_DIM], int layer)
{
#pragma HLS inline off

    FM_TYPE psum[MLP_1_OUT];

#pragma HLS array_partition variable=psum complete
#pragma HLS array_partition variable=mlp_in dim=1 complete
#pragma HLS array_partition variable=mlp_out dim=1 complete
#pragma HLS array_partition variable=mlp_1_weights dim=3 complete
#pragma HLS array_partition variable=mlp_1_bias dim=2 complete
#pragma HLS array_partition variable=mlp_2_weights dim=2 complete
#pragma HLS array_partition variable=mlp_2_bias dim=2 complete


    for(int dim = 0; dim < MLP_2_IN; dim++) {
#pragma HLS pipeline

        // first layer of 300 x 300 VVM
        FM_TYPE sum = mlp_1_bias[layer][dim];
        for(int dim_in1 = 0; dim_in1 < MLP_1_IN; dim_in1++) {
            psum[dim_in1] = mlp_1_weights[layer][dim][dim_in1] * mlp_in[dim_in1];
            sum += psum[dim_in1];
        }
        sum = sum < 0 ? (FM_TYPE)0 : sum;

        // second layer of 300 x 300 VVM
        for(int dim_out2 = 0; dim_out2 < MLP_2_OUT; dim_out2++) {
            mlp_out[dim_out2] += sum * mlp_2_weights[layer][dim_out2][dim];
        }
    }

}


void prepare_mlp_in(FM_TYPE mlp_in[EMB_DIM], int nd, FM_TYPE message_tb[MAX_NODE][EMB_DIM], WT_TYPE _eps)
{
#pragma HLS inline off
    for(int dim = 0; dim < EMB_DIM; dim++) {
        mlp_in[dim] = message_tb[nd][dim] + (1 + _eps) * node_embedding[nd][dim];
    }
}

void prepare_mlp_out(FM_TYPE mlp_out[EMB_DIM], int nd, WT_TYPE* bias)
{
#pragma HLS inline off
    for(int dim = 0; dim < EMB_DIM; dim++) {
        mlp_out[dim] = bias[dim];
    }
}


void update_node_embedding_with_Relu_and_vn_embedding(FM_TYPE mlp_out[EMB_DIM], FM_TYPE emb_vec[EMB_DIM],
       FM_TYPE vn_embedding_read[EMB_DIM], FM_TYPE vn_embedding_write[EMB_DIM], int nd, int layer)
{
#pragma HLS inline off
    for(int dim = 0; dim < EMB_DIM; dim++) {
        if( mlp_out[dim] < 0 && layer != LAYER_NUM - 1 ) {
            mlp_out[dim] = 0;
        }
        
        // Update node embedding with virtual node embedding before message passing
        // Note: Updating VN is not required for the final layer
        if(layer != LAYER_NUM - 1)
        {
            node_embedding[nd][dim] = mlp_out[dim] + vn_embedding_read[dim];
            
            // Aggregate virtual node embedding with node embedding for next layer
            vn_embedding_write[dim] += node_embedding[nd][dim];
        }
        else
        {
            node_embedding[nd][dim] = mlp_out[dim];
        }

        // Store node embedding into a vector for message passing
        emb_vec[dim] = node_embedding[nd][dim];
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

int get_ed_emb_addr(int ef, int layer)
{
    int addr = 0;
    for(int i = 0; i < ef; i++) {
        addr += ed_feature_table[i];
    }
    return addr + layer * (5+6+2);
}


void message_passing_one_node(int nd, FM_TYPE message_tb[MAX_NODE][EMB_DIM], int edge_attr[MAX_EDGE][EDGE_ATTR], int layer)
{
#pragma HLS inline off

#pragma HLS array_partition variable=edge_embedding_table complete
#pragma HLS array_partition variable=edge_attr dim=2 complete


    ////////////// Embedding: compute edge embedding

    int u = nd;
    int total_neigh = degree_table[u * 3];
    int start_idx = degree_table[u * 3 + 1];        

    for(int i = 0; i < total_neigh; i++) {
#pragma HLS loop_tripcount min=1 max=5 avg=3

        int v = neighbor_table[start_idx + i * 2];
        int e = neighbor_table[start_idx + i * 2 + 1];

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
            message_tb[v][dim] += msg;   
        }
    }   
}

void message_passing_one_node_vec(FM_TYPE emb_vec[EMB_DIM], int nd, FM_TYPE message_tb[MAX_NODE][EMB_DIM], int edge_attr[MAX_EDGE][EDGE_ATTR], int layer)
{
#pragma HLS inline off

#pragma HLS array_partition variable=edge_embedding_table complete
#pragma HLS array_partition variable=edge_attr dim=2 complete


    ////////////// Embedding: compute edge embedding

    int u = nd;
    int total_neigh = degree_table[u * 3];
    int start_idx = degree_table[u * 3 + 1];        

    for(int i = 0; i < total_neigh; i++) {
#pragma HLS loop_tripcount min=1 max=5 avg=3

        int v = neighbor_table[start_idx + i * 2];
        int e = neighbor_table[start_idx + i * 2 + 1];

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
            FM_TYPE msg = edge_embed + emb_vec[dim];
            if(msg < 0) msg = 0.0;
            message_tb[v][dim] += msg;   
        }
    }   
}

void clear_message_table_one_node(FM_TYPE message_tb[EMB_DIM], int nd)
{
#pragma HLS inline off
    
    for(int dim = 0; dim < EMB_DIM; dim++) {
        message_tb[dim] = 0;
    }
}

void clear_message_table(FM_TYPE message_tb[MAX_NODE][EMB_DIM], int num_of_nodes)
{
#pragma HLS inline off
    for(int n = 0; n < num_of_nodes; n++) {
        clear_message_table_one_node(message_tb[n], n);
    }
}

void clear_node_embedding(FM_TYPE* node_emb_vec)
{
#pragma HLS inline off

    for(int dim = 0; dim < EMB_DIM; dim++) {
        node_emb_vec[dim] = 0;
    }
}

void one_node_embedding(int nd, int* node_features, FM_TYPE emb_vec[EMB_DIM], FM_TYPE vn_embedding_read[EMB_DIM], FM_TYPE vn_embedding_write[EMB_DIM])
{
#pragma HLS inline off
#pragma HLS array_partition variable=node_embedding_table dim=1 complete

    for(int dim = 0; dim < EMB_DIM; dim++) {
#pragma HLS pipeline        

        FM_TYPE sum = 0;
        for(int nf = 0; nf < ND_FEATURE; nf++) {
            int nd_f = node_features[nd * ND_FEATURE + nf];
            int emb_addr = get_nd_emb_addr(nf);

            FM_TYPE emb_value = node_embedding_table[emb_addr + nd_f][dim]; 
            sum += emb_value;
        }
        // Update node embedding with virtual node before message passing
        node_embedding[nd][dim] = sum + vn_embedding_read[dim];
        emb_vec[dim] = sum + vn_embedding_read[dim];

        // Update virtual node embedding for virtual node MLP
        // This is required for message passing in the next layer
        vn_embedding_write[dim] += emb_vec[dim];
    }
}

void compute_node_embedding(FM_TYPE vn_embedding_read[EMB_DIM], FM_TYPE vn_embedding_write[EMB_DIM], int num_of_nodes, int num_of_edges, int* node_features)
{
#pragma HLS inline off
    ////////////// Embedding: compute input node embedding
    
    int layer = 0;

    FM_TYPE emb_vec1[EMB_DIM];
    FM_TYPE emb_vec2[EMB_DIM];

    one_node_embedding(0, node_features, emb_vec1, vn_embedding_read, vn_embedding_write);
    
    loop_node_emb: for(int nd = 1; nd < num_of_nodes; nd++) {
        if( nd % 2 == 1 ) {
            message_passing_one_node_vec(emb_vec1, nd-1, message1, edge_attr, layer);
            one_node_embedding(nd, node_features, emb_vec2, vn_embedding_read, vn_embedding_write);
        }
        else {
            message_passing_one_node_vec(emb_vec2, nd-1, message1, edge_attr, layer);
            one_node_embedding(nd, node_features, emb_vec1, vn_embedding_read, vn_embedding_write);
        }
    }
    if( (num_of_nodes-1) % 2 == 0 )
        message_passing_one_node_vec(emb_vec1, num_of_nodes-1, message1, edge_attr, layer);
    else
        message_passing_one_node_vec(emb_vec2, num_of_nodes-1, message1, edge_attr, layer);


#ifdef _PRINT_
    printf("\nInitial node embedding after addition of VN:\n");
    for(int nd = 0; nd < 5; nd++) {
        printf("Node %d: ", nd);
        for(int dim = 0; dim < 10; dim++) {
            printf("%.5f ", node_embedding[nd][dim].to_float());
        }
        printf("...\n");
    }

    // Print VN embedding
    printf("\nCompute_node_embedding: Virtual node embedding after aggregate:\n");
    for(int dim = 0; dim < 10; dim++) {
        printf("%.5f ", vn_embedding_write[dim].to_float());
    }
#endif
}

// Perform MLP for virtual node.
// This is to be done before MLP for regular nodes
void virtualnode_MLP(FM_TYPE vn_embedding1[EMB_DIM], FM_TYPE vn_embedding2[EMB_DIM], int layer)
{
#pragma HLS inline off

    // Changed from MLP_1_OUT
    FM_TYPE psum[VN_MLP_1_IN];

#pragma HLS array_partition variable=psum complete
#pragma HLS array_partition variable=vn_embedding2 dim=1 complete
#pragma HLS array_partition variable=vn_embedding1 dim=1 complete
#pragma HLS array_partition variable=virtualnode_mlp_1_weights dim=3 complete
// #pragma HLS array_partition variable=virtualnode_mlp_1_bias dim=2 complete
#pragma HLS array_partition variable=virtualnode_mlp_2_weights dim=2 complete
// #pragma HLS array_partition variable=virtualnode_mlp_2_bias dim=2 complete

    // Initialize virtual node MLP output with bias
    // VN1 will be used to hold the updated VN embedding for the next layer 
    for(int dim = 0; dim < VN_MLP_2_OUT; dim++) {
        #pragma HLS unroll
        vn_embedding1[dim] = virtualnode_mlp_2_bias[layer][dim];
    }

    // Perform both layers of MLP
    for(int dim = 0; dim < VN_MLP_2_IN; dim++) {
#pragma HLS pipeline

        // Perform MLP for hidden layer
        FM_TYPE sum = virtualnode_mlp_1_bias[layer][dim];
        for(int dim_in1 = 0; dim_in1 < VN_MLP_1_IN; dim_in1++) {
            psum[dim_in1] = virtualnode_mlp_1_weights[layer][dim][dim_in1] * vn_embedding2[dim_in1];
            sum += psum[dim_in1];
        }
        sum = sum < 0 ? (FM_TYPE)0 : sum;

        // Perform MLP for output layer
        for(int dim_out2 = 0; dim_out2 < VN_MLP_2_OUT; dim_out2++) {
            vn_embedding1[dim_out2] += sum * virtualnode_mlp_2_weights[layer][dim_out2][dim];
        }
    }

    // Perform ReLU on virtual node MLP output
    for(int dim = 0; dim < VN_MLP_2_OUT; dim++) {
        #pragma HLS unroll
        if(vn_embedding1[dim] < (FM_TYPE)0)
        {
            vn_embedding1[dim] = (FM_TYPE)0;
        }

        // VN2 will be used to aggregate (preparation for the next layer)
        vn_embedding2[dim] = vn_embedding1[dim];
    }

#ifdef _PRINT_
    printf("\nVirtual Node Embedding after MLP:\n");
    for(int dim = 0; dim < 10; dim++) {
        printf("%.5f ", vn_embedding2[dim].to_float());
    }
#endif
}

void MLP_wrapper(FM_TYPE mlp_in[EMB_DIM], FM_TYPE mlp_out[EMB_DIM], int nd, FM_TYPE emb_vec[EMB_DIM], FM_TYPE message_tb[MAX_NODE][EMB_DIM], 
        FM_TYPE vn_embedding_read[EMB_DIM], FM_TYPE vn_embedding_write[EMB_DIM],
        FM_TYPE _eps, int layer)
{
#pragma HLS inline off

    prepare_mlp_in(mlp_in, nd, message_tb, _eps);
    prepare_mlp_out(mlp_out, nd, mlp_2_bias[layer]);
    MLP_one_node(nd, mlp_in, mlp_out, layer);
    update_node_embedding_with_Relu_and_vn_embedding(mlp_out, emb_vec, vn_embedding_read, vn_embedding_write, nd, layer);
}

void compute_CONV_layer(FM_TYPE vn_embedding_read[EMB_DIM], FM_TYPE vn_embedding_write[EMB_DIM], int num_of_nodes, int num_of_edges, int layer)
{
#pragma HLS inline off
    

    FM_TYPE mlp_in[EMB_DIM];
    FM_TYPE mlp_out[EMB_DIM];

    FM_TYPE emb_vec1[EMB_DIM];
    FM_TYPE emb_vec2[EMB_DIM];

    /// something special in GIN
    WT_TYPE _eps = mlp_eps[layer];

    if( layer % 2 == 0) {
        clear_message_table(message2, num_of_nodes);
        MLP_wrapper(mlp_in, mlp_out, 0, emb_vec1, message1, vn_embedding_read, vn_embedding_write, _eps, layer);
    }
    else {
        clear_message_table(message1, num_of_nodes);
        MLP_wrapper(mlp_in, mlp_out, 0, emb_vec1, message2, vn_embedding_read, vn_embedding_write, _eps, layer);
    }

    loop_compute_conv: for(int nd = 1; nd < num_of_nodes; nd++) {
        if( nd % 2 == 1 && layer % 2 == 0) {
            message_passing_one_node_vec(emb_vec1, nd-1, message2, edge_attr, layer + 1);
            MLP_wrapper(mlp_in, mlp_out, nd, emb_vec2, message1, vn_embedding_read, vn_embedding_write, _eps, layer);
        }
        else if(nd % 2 == 1 && layer % 2 == 1) {
            message_passing_one_node_vec(emb_vec1, nd-1, message1, edge_attr, layer + 1);
            MLP_wrapper(mlp_in, mlp_out, nd, emb_vec2, message2, vn_embedding_read, vn_embedding_write, _eps, layer);
        }
        else if(nd % 2 == 0 && layer % 2 == 0) {
            message_passing_one_node_vec(emb_vec2, nd-1, message2, edge_attr, layer + 1);
            MLP_wrapper(mlp_in, mlp_out, nd, emb_vec1, message1, vn_embedding_read, vn_embedding_write, _eps, layer);
        }
        else { // if(nd % 2 == 0 && layer % 2 == 1) {
            message_passing_one_node_vec(emb_vec2, nd-1, message1, edge_attr, layer + 1);
            MLP_wrapper(mlp_in, mlp_out, nd, emb_vec1, message2, vn_embedding_read, vn_embedding_write, _eps, layer);
        }
    }

    if( (num_of_nodes-1) % 2 == 0 && layer % 2 == 0)
        message_passing_one_node_vec(emb_vec1, num_of_nodes-1, message2, edge_attr, layer + 1);
    else if( (num_of_nodes-1) % 2 == 0 && layer % 2 == 1)
        message_passing_one_node_vec(emb_vec1, num_of_nodes-1, message1, edge_attr, layer + 1);
    else if( (num_of_nodes-1) % 2 == 1 && layer % 2 == 0)
        message_passing_one_node_vec(emb_vec2, num_of_nodes-1, message2, edge_attr, layer + 1);
    else if( (num_of_nodes-1) % 2 == 1 && layer % 2 == 1)
        message_passing_one_node_vec(emb_vec2, num_of_nodes-1, message1, edge_attr, layer + 1);
    

#ifdef _PRINT_
    printf("\nOutput of Conv %d\n", layer);
    for(int nd = 0; nd < 5; nd++) {
        printf("Node %d: ", nd);
        for(int dim = 0; dim < 10; dim++) {
            printf("%.5f ", node_embedding[nd][dim].to_float());
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

// Load virtual node embeddings from DRAM to BRAM
void load_virtualnode_embedding(FM_TYPE* gnn_node_virtualnode_embedding_weight_fixed)
{
#pragma HLS inline off
    for(int dim = 0; dim < EMB_DIM; dim++) 
    {
        virtualnode_embedding_weight[dim] = gnn_node_virtualnode_embedding_weight_fixed[dim];
    }
}

// Initialize virtual node embeddings
void initialize_virtualnode_embedding(FM_TYPE vn_embedding1[EMB_DIM], FM_TYPE vn_embedding2[EMB_DIM])
{
#pragma HLS inline off
#pragma HLS array_partition variable=vn_embedding1 complete 
#pragma HLS array_partition variable=vn_embedding2 complete 
#pragma HLS array_partition variable=virtualnode_embedding_weight complete 

    for(int dim = 0; dim < EMB_DIM; dim++) 
    {
        FM_TYPE vn_embedding_temp = virtualnode_embedding_weight[dim];
        vn_embedding1[dim] = vn_embedding_temp;
        vn_embedding2[dim] = vn_embedding_temp;
    }
#ifdef _PRINT_
    printf("\nInitial virtual node embedding:\n");
    for(int dim = 0; dim < 10; dim++) {
        printf("%.5f ", vn_embedding1[dim].to_float());
    }
#endif
}

void prepare_degree_neighbor_table(int* edge_list, int num_of_nodes, int num_of_edges)
{
#pragma HLS inline off

    for(int n = 0; n < num_of_nodes * 3; n++) {
        degree_table[n] = 0;
    }
    
    for(int e = 0; e < num_of_edges; e++) {
        int u = edge_list[e * 2];     // source node id
        int v = edge_list[e * 2 + 1];   // target node id

        degree_table[u * 3] += 1;
    }

    for(int n = 1; n < num_of_nodes; n++) {
        degree_table[n * 3 + 1] = degree_table[(n - 1) * 3] * 2 + degree_table[(n - 1) * 3 + 1]; // *2: one is for the neighbor node id; the next one is for the corresponding edge id
    }

    for(int e = 0; e < num_of_edges; e++) {
        int u = edge_list[e * 2];     // source node id
        int v = edge_list[e * 2 + 1];   // target node id

        int total_neigh = degree_table[u * 3];
        int start_idx = degree_table[u * 3 + 1];
        int offset_idx = degree_table[u * 3 + 2];
        
        neighbor_table[start_idx + offset_idx] = v;
        neighbor_table[start_idx + offset_idx + 1] = e;
        degree_table[u * 3 + 2] += 2;
    }

#ifdef _PRINT_
    printf("Degree Table:\n");
    for(int n = 0; n < num_of_nodes; n++) {
        printf("Node %d's degree: %d\t", n, degree_table[n * 3]);
        printf("Node %d's start_idx: %d\t", n, degree_table[n * 3 + 1]);
        printf("Node %d's offset_idx: %d\n", n, degree_table[n * 3 + 2]);
    }

    printf("Neighbor Table:\n");
    for(int n = 0; n < num_of_nodes; n++) {
        int total_neigh = degree_table[n * 3];
        int start_idx = degree_table[n * 3 + 1];
        int offset_idx = degree_table[n * 3 + 2];

        printf("Node %d's neighbors:\n", n);
        for(int i = 0; i < total_neigh; i++) {
            printf("%d ", neighbor_table[start_idx + i * 2]);
        }
        printf("\n");
    }
#endif

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
    WT_TYPE gnn_node_virtualnode_mlp_2_bias_fixed[VN_LAYER_NUM * VN_MLP_2_OUT]
    )
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
#pragma HLS bind_storage variable=graph_embedding type=RAM_2P impl=bram
#pragma HLS bind_storage variable=node_embedding type=RAM_2P impl=bram
#pragma HLS bind_storage variable=node_embedding_table type=RAM_2P impl=bram
#pragma HLS bind_storage variable=edge_embedding_table type=RAM_2P impl=bram
#pragma HLS bind_storage variable=message1 type=RAM_2P impl=uram
#pragma HLS bind_storage variable=message2 type=RAM_2P impl=uram
#pragma HLS bind_storage variable=virtualnode_embedding_weight type=RAM_2P impl=bram
// #pragma HLS bind_storage variable=vn_embedding1 type=RAM_2P impl=bram
// #pragma HLS bind_storage variable=vn_embedding2 type=RAM_2P impl=bram

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

    ////////////// Preprocess: prepare degree table and neighbor table
    prepare_degree_neighbor_table(edge_list, num_of_nodes, num_of_edges);

    ///////////// clear message table
    clear_message_table(message1, num_of_nodes);
    clear_message_table(message2, num_of_nodes);

    // Initialize Virtual Node Embeddings
    // VN1: Always used for message passing in the current layer
    // VN2: Always used for aggregation (preparation) for the next layer
    initialize_virtualnode_embedding(vn_embedding1, vn_embedding2);

    ////////////// Embedding: compute input node embedding
    compute_node_embedding(vn_embedding1, vn_embedding2, num_of_nodes, num_of_edges, node_feature);

    ////////////// CONV layers //////////////////////////////////
    for(int layer = 0; layer < VN_LAYER_NUM; layer++) {
        virtualnode_MLP(vn_embedding1, vn_embedding2, layer);
        compute_CONV_layer(vn_embedding1, vn_embedding2, num_of_nodes, num_of_edges, layer);
    }

    compute_CONV_layer(vn_embedding1, vn_embedding2, num_of_nodes, num_of_edges, LAYER_NUM - 1);

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
