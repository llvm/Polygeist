#include "load_inputs.h"

using std::array;

static const int nd_feature_offsets[ND_FEATURE] = {0, 119, 123, 135, 147, 157, 163, 169, 171};

void load_weights(
    WT_TYPE node_mlp_1_weights_in[NUM_LAYERS][MLP_1_OUT][EMB_DIM],
    WT_TYPE node_mlp_1_bias_in[NUM_LAYERS][MLP_1_OUT],
    WT_TYPE node_mlp_2_weights_in[NUM_LAYERS][EMB_DIM][MLP_1_OUT],
    WT_TYPE node_mlp_2_bias_in[NUM_LAYERS][EMB_DIM],
    WT_TYPE edge_embedding_weight_in[NUM_LAYERS][ED_FEATURE_PER_LAYER][EMB_DIM],
    WT_TYPE graph_pred_weights_in[NUM_TASK][EMB_DIM],
    WT_TYPE graph_pred_bias_in[NUM_TASK]
)
{
#pragma HLS INLINE off

    load_mlp_1_bias: for (int l = 0; l < NUM_LAYERS; l++)
    {
        load_mlp_1_bias_dim: for (int dim = 0; dim < MLP_1_OUT; dim++)
        {
            node_mlp_1_bias[l][dim] = node_mlp_1_bias_in[l][dim];
        }
    }

    load_mlp_2_bias: for (int l = 0; l < NUM_LAYERS; l++)
    {
        load_mlp_2_bias_dim: for (int dim = 0; dim < EMB_DIM; dim++)
        {
            node_mlp_2_bias[l][dim] = node_mlp_2_bias_in[l][dim];
        }
    }

    load_mlp_1_weights: for (int l = 0; l < NUM_LAYERS; l++)
    {
        load_mlp_1_weights_dim_out: for (int dim_out = 0; dim_out < MLP_1_OUT; dim_out++)
        {
            load_mlp_1_weights_dim_in: for (int dim_in = 0; dim_in < EMB_DIM; dim_in++)
            {
                node_mlp_1_weights[l][dim_out][dim_in] = node_mlp_1_weights_in[l][dim_out][dim_in];
            }
        }
    }

    load_mlp_2_weights: for (int l = 0; l < NUM_LAYERS; l++)
    {
        load_mlp_2_weights_dim_out: for (int dim_out = 0; dim_out < EMB_DIM; dim_out++)
        {
            load_mlp_2_weights_dim_in: for (int dim_in = 0; dim_in < MLP_1_OUT; dim_in++)
            {
                node_mlp_2_weights[l][dim_out][dim_in] = node_mlp_2_weights_in[l][dim_out][dim_in];
            }
        }
    }

    load_edge_emb_weights: for (int l = 0; l < NUM_LAYERS; l++)
    {
        load_edge_emb_weights_feat: for (int i = 0; i < ED_FEATURE_PER_LAYER; i++)
        {
            load_edge_emb_weights_dim: for (int dim = 0; dim < EMB_DIM; dim++)
            {
                WT_TYPE tmp = edge_embedding_weight_in[l][i][dim];
                for (int pe_id = 0; pe_id < EDGE_PARALLEL; pe_id++)
                {
#pragma HLS UNROLL
                    edge_embedding_weights[pe_id][l][i][dim] = tmp;
                }
            }
        }
    }

    load_graph_pred_bias: for (int t = 0; t < NUM_TASK; t++)
    {
        graph_pred_bias[t] = graph_pred_bias_in[t];
    }

    load_graph_pred_weights: for (int t = 0; t < NUM_TASK; t++)
    {
        load_graph_pred_weights_dim: for (int dim_in = 0; dim_in < EMB_DIM; dim_in++)
        {
            graph_pred_weights[t][dim_in] = graph_pred_weights_in[t][dim_in];
        }
    }
}

void load_graph(
    edge_t* edge_list_in,
    edge_attr_t* edge_attr_in,
    int num_of_nodes,
    int num_of_edges
)
{
#pragma HLS INLINE off

    WT_TYPE degree_inv_sqrt[MAX_NODE];
    int full_pe_degree_tables[EDGE_PARALLEL][MAX_NODE];
    int neighbor_table_offsets[MAX_NODE];
    int neighbor_tables_offsets[EDGE_PARALLEL][MAX_NODE];

#pragma HLS ARRAY_PARTITION variable=degree_tables complete dim=1
#pragma HLS AGGREGATE variable=degree_tables
#pragma HLS ARRAY_RESHAPE variable=full_pe_degree_tables complete dim=1
#pragma HLS ARRAY_PARTITION variable=neighbor_tables complete dim=1
#pragma HLS ARRAY_PARTITION variable=neighbor_tables_offsets complete dim=1
#pragma HLS ARRAY_PARTITION variable=edge_attrs complete dim=1
#pragma HLS ARRAY_PARTITION variable=num_of_edges_per_pe complete dim=1

    for (int i = 0; i < num_of_nodes; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODES max=ANALYSIS_MAX_NODES avg=ANALYSIS_AVG_NODES
        degree_table[i] = 0;
        degree_inv_sqrt[i] = 0;

        for (int j = 0; j < EDGE_PARALLEL; j++)
        {
#pragma HLS UNROLL
            full_pe_degree_tables[j][i] = 0;
        }
    }

    for (int i = 0; i < num_of_edges; i++)
    {
        // TODO: can we make this II=1?
#pragma HLS PIPELINE II=3
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_EDGES max=ANALYSIS_MAX_EDGES avg=ANALYSIS_AVG_EDGES
        edge_t edge = edge_list_in[i];
        int u = edge.u;
        int v = edge.v;
        int pe_id = v % EDGE_PARALLEL;
        degree_table[u]++;
        full_pe_degree_tables[pe_id][u]++;
        degree_inv_sqrt[u] = hls::recip(hls::sqrt(WT_TYPE(degree_table[u] + 1)));
    }

    int acc = 0;
    int num_of_nodes_per_pe[EDGE_PARALLEL];
    for (int i = 0; i < EDGE_PARALLEL; i++)
    {
#pragma HLS UNROLL
        num_of_nodes_per_pe[i] = 0;
        num_of_edges_per_pe[i] = 0;
    }

    for (int i = 0; i < num_of_nodes; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODES max=ANALYSIS_MAX_NODES avg=ANALYSIS_AVG_NODES
        int degree = degree_table[i];
        neighbor_table_offsets[i] = acc;
        acc += degree;

        ap_int<EDGE_PARALLEL> node_pes = 0;
        for (int j = 0; j < EDGE_PARALLEL; j++)
        {
#pragma HLS UNROLL
            int degree_j = full_pe_degree_tables[j][i];
            neighbor_tables_offsets[j][i] = num_of_edges_per_pe[j];
            num_of_edges_per_pe[j] += degree_j;

            if (degree_j > 0)
            {
                int pe_num_of_nodes = num_of_nodes_per_pe[j];
                degree_tables[j][pe_num_of_nodes] = node_t {i, degree_j};
                num_of_nodes_per_pe[j] = pe_num_of_nodes + 1;
                node_pes[j] = 1;
            }
        }
        pes_per_node[i] = node_pes;
    }

    for (int i = 0; i < num_of_edges; i++)
    {
        // TODO: can we make this II=1?
#pragma HLS PIPELINE II=4
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_EDGES max=ANALYSIS_MAX_EDGES avg=ANALYSIS_AVG_EDGES
        edge_t edge = edge_list_in[i];
        int u = edge.u;
        int v = edge.v;
        int pe_id = v % EDGE_PARALLEL;
        int e = neighbor_table_offsets[u];
        int e_pe = neighbor_tables_offsets[pe_id][u];
        neighbor_table_offsets[u] = e + 1;
        neighbor_tables[pe_id][e_pe] = v / EDGE_PARALLEL;
        neighbor_tables_offsets[pe_id][u] = e_pe + 1;
        edge_attrs[pe_id][e_pe] = edge_attr_in[i];
    }
}

void load_input_node_embeddings(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    node_feature_t* node_feature,
    WT_TYPE node_embedding_weight[ND_FEATURE_TOTAL][EMB_DIM],
    FM_TYPE messages[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    int num_of_nodes
)
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=nd_feature_offsets complete dim=1

    /*Embedding: compute input node embedding */
    for (int nd = 0; nd < num_of_nodes; nd++)
    {
#pragma HLS PIPELINE II=ceildiv(EMB_DIM, APPLY_PARALLEL)
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODES max=ANALYSIS_MAX_NODES avg=ANALYSIS_AVG_NODES

        array<WT_TYPE, EMB_DIM> weights[ND_FEATURE];
        node_feature_t node_feature_nd = node_feature[nd];
        for (int nf = 0; nf < ND_FEATURE; nf++)
        {
#pragma HLS UNROLL
            int nd_f = nd_feature_offsets[nf] + node_feature_nd[nf];
            weights[nf] = *((array<WT_TYPE, EMB_DIM>*)node_embedding_weight[nd_f]);
        }

        for (int dim_base = 0; dim_base < EMB_DIM; dim_base += APPLY_PARALLEL)
        {
            ne_out_t embedding;
            for (int dim_offset = 0; dim_offset < APPLY_PARALLEL; dim_offset++)
            {
                int dim = dim_base + dim_offset;
                FM_TYPE h_node_nd_dim = 0;
                for (int nf = 0; nf < ND_FEATURE; nf++)
                {
                    h_node_nd_dim += weights[nf][dim];
                }
                h_node[nd][dim] = h_node_nd_dim;
                embedding[dim_offset] = h_node_nd_dim;

                // in preparation for next round of message passing
                messages[nd % EDGE_PARALLEL][nd / EDGE_PARALLEL][dim] = 0;
            }
            embeddings[nd % NODE_PARALLEL] << embedding;
        }
    }
}
