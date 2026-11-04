#include "load_inputs.h"

using std::array;

static const int nd_feature_offsets[ND_FEATURE] = {0, 119, 123, 135, 147, 157, 163, 169, 171};

void load_weights(
    WT_TYPE convs_weight_in[NUM_LAYERS][100][100],
    WT_TYPE convs_bias_in[NUM_LAYERS][100],
    WT_TYPE convs_root_emb_weight_in[NUM_LAYERS][100],
    WT_TYPE bn_weight_in[NUM_LAYERS][100],
    WT_TYPE bn_bias_in[NUM_LAYERS][100],
    WT_TYPE bn_mean_in[NUM_LAYERS][100],
    WT_TYPE bn_var_in[NUM_LAYERS][100],
    WT_TYPE edge_embedding_weight_in[NUM_LAYERS][ED_FEATURE_PER_LAYER][EMB_DIM],
    WT_TYPE graph_pred_weights_in[NUM_TASK][EMB_DIM],
    WT_TYPE graph_pred_bias_in[NUM_TASK]
)
{
#pragma HLS INLINE off

    load_layer_weights: for (int l = 0; l < NUM_LAYERS; l++)
    {
        load_layer_weights_dim: for (int dim = 0; dim < EMB_DIM; dim++)
        {
#pragma HLS PIPELINE II=6
            convs_bias[l][dim] = convs_bias_in[l][dim];
            convs_root_emb_weight[l][dim] = convs_root_emb_weight_in[l][dim];
            bn_weight[l][dim] = bn_weight_in[l][dim];
            bn_bias[l][dim] = bn_bias_in[l][dim];
            bn_mean[l][dim] = bn_mean_in[l][dim];
            bn_sqrt_var[l][dim] = hls::sqrt(bn_var_in[l][dim] + ap_fixed_epsilon<WT_TYPE>());
        }
    }

    load_conv_weights: for (int l = 0; l < NUM_LAYERS; l++)
    {
        load_conv_weights_dim_out: for (int dim_out = 0; dim_out < EMB_DIM; dim_out++)
        {
            load_conv_weights_dim_in: for (int dim_in = 0; dim_in < EMB_DIM; dim_in++)
            {
                convs_weight[l][dim_out][dim_in] = convs_weight_in[l][dim_out][dim_in];
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
    int neighbor_table_offsets[MAX_NODE];
    int neighbor_tables_offsets[EDGE_PARALLEL][MAX_NODE];

#pragma HLS ARRAY_PARTITION variable=degree_tables complete dim=1
#pragma HLS ARRAY_PARTITION variable=degree_tables complete dim=3
#pragma HLS ARRAY_PARTITION variable=neighbor_tables complete dim=1
#pragma HLS ARRAY_PARTITION variable=neighbor_tables_offsets complete dim=1
#pragma HLS ARRAY_PARTITION variable=norms complete dim=1
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
            degree_tables[j][i][0] = 0;
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
        degree_tables[pe_id][u][0]++;
        degree_inv_sqrt[u] = hls::recip(hls::sqrt(WT_TYPE(degree_table[u] + 1)));
    }

    int acc = 0;
    for (int i = 0; i < EDGE_PARALLEL; i++)
    {
#pragma HLS UNROLL
        num_of_edges_per_pe[i] = 0;
    }

    for (int i = 0; i < num_of_nodes; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODES max=ANALYSIS_MAX_NODES avg=ANALYSIS_AVG_NODES
        int degree = degree_table[i];
        neighbor_table_offsets[i] = acc;
        acc += degree;

        for (int j = 0; j < EDGE_PARALLEL; j++)
        {
#pragma HLS UNROLL
            int degree_j = degree_tables[j][i][0];
            neighbor_tables_offsets[j][i] = num_of_edges_per_pe[j];
            degree_tables[j][i][1] = num_of_edges_per_pe[j];
            num_of_edges_per_pe[j] += degree_j;
        }
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
        norms[pe_id][e_pe] = degree_inv_sqrt[u] * degree_inv_sqrt[v];
        edge_attrs[pe_id][e_pe] = edge_attr_in[i];
    }
}

void load_input_node_embeddings(
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
// I wouldn't pipeline this loop, myself, but Vitis wanted to, so:
#pragma HLS PIPELINE II=ceildiv(EMB_DIM, LOAD_IN_EMB_PARALLEL)
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODES max=ANALYSIS_MAX_NODES avg=ANALYSIS_AVG_NODES

        array<WT_TYPE, EMB_DIM> weights[ND_FEATURE];
        node_feature_t node_feature_nd = node_feature[nd];
        for (int nf = 0; nf < ND_FEATURE; nf++)
        {
#pragma HLS UNROLL
            int nd_f = nd_feature_offsets[nf] + node_feature_nd[nf];
            weights[nf] = *((array<WT_TYPE, EMB_DIM>*)node_embedding_weight[nd_f]);
        }

        ne_out_t embedding;
        int embedding_offset = 0;
        for (int dim_base = 0; dim_base < EMB_DIM; dim_base += LOAD_IN_EMB_PARALLEL)
        {
// Commented out since we are pipelining the outer loop:
// #pragma HLS PIPELINE II=1
            for (int dim_offset = 0; dim_offset < LOAD_IN_EMB_PARALLEL; dim_offset++)
            {
                int dim = dim_base + dim_offset;
                FM_TYPE h_node_nd_dim = 0;
                for (int nf = 0; nf < ND_FEATURE; nf++)
                {
                    h_node_nd_dim += weights[nf][dim];
                }
                h_node[nd][dim] = h_node_nd_dim;

                // in preparation for next round of message passing
                messages[nd % EDGE_PARALLEL][nd / EDGE_PARALLEL][dim] = 0;
            }
        }
    }
}
