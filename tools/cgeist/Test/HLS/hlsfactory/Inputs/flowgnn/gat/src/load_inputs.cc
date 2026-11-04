#include "load_inputs.h"

void load_weights(
    WT_TYPE scoring_fn_target_in[NUM_LAYERS][NUM_HEADS][EMB_DIM],
    WT_TYPE scoring_fn_source_in[NUM_LAYERS][NUM_HEADS][EMB_DIM],
    WT_TYPE linear_proj_weights_in[NUM_LAYERS][NUM_HEADS][EMB_DIM][NUM_HEADS][EMB_DIM],
    WT_TYPE skip_proj_weights_in[NUM_LAYERS][NUM_HEADS][EMB_DIM][NUM_HEADS][EMB_DIM],
    WT_TYPE graph_pred_weights_in[NUM_TASK][EMB_DIM],
    WT_TYPE graph_pred_bias_in[NUM_TASK]
)
{
#pragma HLS INLINE off

    load_scoring_fn_target: for (int l = 0; l < NUM_LAYERS; l++)
    {
        load_scoring_fn_target_head: for (int h = 0; h < NUM_HEADS; h++)
        {
#pragma HLS PIPELINE II=ceildiv(EMB_DIM, 2)
            load_scoring_fn_target_dim: for (int dim = 0; dim < EMB_DIM; dim++)
            {
                scoring_fn_target[l][h][dim] = scoring_fn_target_in[l][h][dim];
            }
        }
    }

    load_scoring_fn_source: for (int l = 0; l < NUM_LAYERS; l++)
    {
        load_scoring_fn_source_head: for (int h = 0; h < NUM_HEADS; h++)
        {
#pragma HLS PIPELINE II=ceildiv(EMB_DIM, 2)
            load_scoring_fn_source_dim: for (int dim = 0; dim < EMB_DIM; dim++)
            {
                scoring_fn_source[l][h][dim] = scoring_fn_source_in[l][h][dim];
            }
        }
    }

    load_linear_proj_weights: for (int l = 0; l < NUM_LAYERS; l++)
    {
        load_linear_proj_weights_head_out: for (int h_out = 0; h_out < NUM_HEADS; h_out++)
        {
            load_linear_proj_weights_dim_out: for (int dim_out = 0; dim_out < EMB_DIM; dim_out++)
            {
                load_linear_proj_weights_head_in: for (int h_in = 0; h_in < NUM_HEADS; h_in++)
                {
#pragma HLS PIPELINE II=ceildiv(EMB_DIM, 2)
                    load_linear_proj_weights_dim_in: for (int dim_in = 0; dim_in < EMB_DIM; dim_in++)
                    {
                        linear_proj_weights[l][h_out][dim_out][h_in][dim_in] = linear_proj_weights_in[l][h_out][dim_out][h_in][dim_in];
                    }
                }
            }
        }
    }

    load_skip_proj_weights: for (int l = 0; l < NUM_LAYERS; l++)
    {
        load_skip_proj_weights_head_out: for (int h_out = 0; h_out < NUM_HEADS; h_out++)
        {
            load_skip_proj_weights_dim_out: for (int dim_out = 0; dim_out < EMB_DIM; dim_out++)
            {
                load_skip_proj_weights_head_in: for (int h_in = 0; h_in < NUM_HEADS; h_in++)
                {
                    load_skip_proj_weights_dim_in: for (int dim_in = 0; dim_in < EMB_DIM; dim_in++)
                    {
                        skip_proj_weights[l][h_out][dim_out][h_in][dim_in] = skip_proj_weights_in[l][h_out][dim_out][h_in][dim_in];
                    }
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
    int num_of_nodes,
    int num_of_edges
)
{
#pragma HLS INLINE off

    int neighbor_tables_offsets[EDGE_PARALLEL][MAX_NODE];

#pragma HLS ARRAY_PARTITION variable=degree_tables complete dim=1
#pragma HLS ARRAY_PARTITION variable=neighbor_tables complete dim=1
#pragma HLS ARRAY_PARTITION variable=neighbor_tables_offsets complete dim=1
#pragma HLS ARRAY_PARTITION variable=num_of_edges_per_pe complete dim=1

    for (int i = 0; i < num_of_nodes; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODES max=ANALYSIS_MAX_NODES avg=ANALYSIS_AVG_NODES
        degree_table[i] = 1;

        for (int j = 0; j < EDGE_PARALLEL; j++)
        {
#pragma HLS UNROLL
            degree_tables[j][i] = (i % EDGE_PARALLEL == j) ? 1 : 0;
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
        int pe_id = u % EDGE_PARALLEL;
        degree_table[v]++;
        degree_tables[pe_id][v]++;
    }

    for (int i = 0; i < EDGE_PARALLEL; i++)
    {
#pragma HLS UNROLL
        num_of_edges_per_pe[i] = 0;
    }

    for (int i = 0; i < num_of_nodes; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODES max=ANALYSIS_MAX_NODES avg=ANALYSIS_AVG_NODES
        for (int j = 0; j < EDGE_PARALLEL; j++)
        {
#pragma HLS UNROLL
            int acc_num_of_edges = num_of_edges_per_pe[j];
            int degree_j = degree_tables[j][i];
            neighbor_tables_offsets[j][i] = acc_num_of_edges;
            num_of_edges_per_pe[j] = acc_num_of_edges + degree_j;

            if (i % EDGE_PARALLEL == j)
            {
                // add self edge for every node
                neighbor_tables[j][acc_num_of_edges] = i / EDGE_PARALLEL;
                neighbor_tables_offsets[j][i] = acc_num_of_edges + 1;
            }
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
        int pe_id = u % EDGE_PARALLEL;
        int e_pe = neighbor_tables_offsets[pe_id][v];
        neighbor_tables[pe_id][e_pe] = u / EDGE_PARALLEL;
        neighbor_tables_offsets[pe_id][v] = e_pe + 1;
    }
}

void load_input_node_embeddings(node_feature_t* node_feature, int num_of_nodes)
{
#pragma HLS INLINE off

    /*Embedding: compute input node embedding */
    for (int nd = 0; nd < num_of_nodes; nd++)
    {
#pragma HLS PIPELINE II=ceildiv(EMB_DIM, APPLY_PARALLEL)
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODES max=ANALYSIS_MAX_NODES avg=ANALYSIS_AVG_NODES

        node_feature_t node_feature_nd = node_feature[nd];
        FM_VEC nodes_features_proj[EMB_DIM];
#pragma HLS ARRAY_PARTITION variable=nodes_features_proj complete dim=0

        for (int dim = 0; dim < EMB_DIM; dim++)
        {
            out_nodes_features_skip_concat_bias_ping[nd][dim] = FM_TYPE(0);
            nodes_features_proj[dim] = FM_TYPE(0);
        }

        for (int nf = 0; nf < ND_FEATURE; nf++)
        {
            FM_TYPE node_feature_nd_nf = node_feature_nd[nf];
            out_nodes_features_skip_concat_bias_ping[nd][nf][0] = node_feature_nd_nf;

            for (int dim_out = 0; dim_out < EMB_DIM; dim_out++)
            {
                WT_VEC weights;
                for (int head_out = 0; head_out < NUM_HEADS; head_out++)
                {
                    weights[head_out] = linear_proj_weights[0][head_out][dim_out][0][nf];
                }
                nodes_features_proj[dim_out] += node_feature_nd_nf * weights;
            }
        }

        FM_VEC scores_source_acc = FM_TYPE(0);
        FM_VEC scores_target_acc = FM_TYPE(0);
        for (int dim = 0; dim < EMB_DIM; dim++)
        {
            WT_VEC scoring_fn_source_weights;
            WT_VEC scoring_fn_target_weights;
            for (int head = 0; head < NUM_HEADS; head++)
            {
                scoring_fn_source_weights[head] = scoring_fn_source[0][head][dim];
                scoring_fn_target_weights[head] = scoring_fn_target[0][head][dim];
            }

            FM_VEC result = nodes_features_proj[dim];
            h_node_ping[nd % EDGE_PARALLEL][nd / EDGE_PARALLEL][dim] = result;
            scores_source_acc += result * scoring_fn_source_weights;
            scores_target_acc += result * scoring_fn_target_weights;
        }
        for (int pe_id = 0; pe_id < EDGE_PARALLEL; pe_id++)
        {
            scores_source_ping[pe_id][nd] = scores_source_acc;
        }
        scores_target_ping[nd % EDGE_PARALLEL][nd / EDGE_PARALLEL] = scores_target_acc;
    }
}
