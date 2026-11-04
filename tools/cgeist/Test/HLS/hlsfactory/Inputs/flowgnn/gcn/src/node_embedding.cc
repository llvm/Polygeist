#include "node_embedding.h"

// #region Internal Function Declarations
static void accumulate(
    FM_TYPE message[EMB_DIM],
    FM_TYPE h_node_v[EMB_DIM],
    FM_TYPE accs[EMB_DIM],
    int degree,
    int layer_num,
    int dim_base
);
static void output(
    FM_TYPE h_node_v[EMB_DIM],
    FM_TYPE accs[EMB_DIM],
    hls::stream<ne_out_t>& embeddings,
    int dim_base
);
// #endregion

void node_embedding_multi_pe(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    FM_TYPE message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    int layer_num,
    int num_of_nodes
)
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=degree_table cyclic factor=NODE_PARALLEL dim=1
#pragma HLS ARRAY_PARTITION variable=h_node cyclic factor=NODE_PARALLEL dim=1
#pragma HLS ARRAY_PARTITION variable=h_node cyclic factor=APPLY_PARALLEL dim=2

    FM_TYPE accs_ping[NODE_PARALLEL][EMB_DIM];
#pragma HLS ARRAY_PARTITION variable=accs_ping complete dim=1
#pragma HLS ARRAY_PARTITION variable=accs_ping complete dim=2
    FM_TYPE accs_pong[NODE_PARALLEL][EMB_DIM];
#pragma HLS ARRAY_PARTITION variable=accs_pong complete dim=1
#pragma HLS ARRAY_PARTITION variable=accs_pong complete dim=2

    int num_iters = ceildiv(num_of_nodes, NODE_PARALLEL) + 1;
    for (
        int i = 0, acc_v_base = 0, out_v_base = -NODE_PARALLEL;
        i < num_iters;
        i++, acc_v_base += NODE_PARALLEL, out_v_base += NODE_PARALLEL
    )
    {
#pragma HLS LOOP_TRIPCOUNT min=(ceildiv(ANALYSIS_MIN_NODES, NODE_PARALLEL) + 1) max=(ceildiv(ANALYSIS_MAX_NODES, NODE_PARALLEL) + 1) avg=(ceildiv(ANALYSIS_AVG_NODES, NODE_PARALLEL) + 1)
        for (int dim_base = 0; dim_base < EMB_DIM; dim_base += APPLY_PARALLEL)
        {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=message inter false

            if (i != 0)
            {
                for (int v_offset = 0; v_offset < NODE_PARALLEL; v_offset++)
                {
#pragma HLS UNROLL
                    int v = out_v_base + v_offset;
                    if (v < num_of_nodes)
                    {
                        output(
                            h_node[v],
                            (i % 2 == 0) ? accs_pong[v_offset] : accs_ping[v_offset],
                            embeddings[v_offset],
                            dim_base
                        );
                    }
                }
            }

            if (i != num_iters - 1)
            {
                for (int v_offset = 0; v_offset < NODE_PARALLEL; v_offset++)
                {
#pragma HLS UNROLL
                    int v = acc_v_base + v_offset;
                    if (v < num_of_nodes)
                    {
                        accumulate(
                            message[v % EDGE_PARALLEL][v / EDGE_PARALLEL],
                            h_node[v],
                            (i % 2 == 0) ? accs_ping[v_offset] : accs_pong[v_offset],
                            degree_table[v],
                            layer_num,
                            dim_base
                        );
                    }
                }
            }
        }
    }
}

static void accumulate(
    FM_TYPE message[EMB_DIM],
    FM_TYPE h_node_v[EMB_DIM],
    FM_TYPE accs[EMB_DIM],
    int degree,
    int layer_num,
    int dim_base
)
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=convs_weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=convs_weight cyclic factor=APPLY_PARALLEL dim=3
#pragma HLS ARRAY_PARTITION variable=convs_bias complete dim=2
#pragma HLS ARRAY_PARTITION variable=bn_sqrt_var cyclic factor=APPLY_PARALLEL dim=2
#pragma HLS ARRAY_PARTITION variable=bn_weight cyclic factor=APPLY_PARALLEL dim=2
#pragma HLS ARRAY_PARTITION variable=bn_mean cyclic factor=APPLY_PARALLEL dim=2
#pragma HLS ARRAY_PARTITION variable=bn_bias cyclic factor=APPLY_PARALLEL dim=2
#pragma HLS ARRAY_PARTITION variable=convs_root_emb_weight cyclic factor=APPLY_PARALLEL dim=2

    for (int dim_offset = 0; dim_offset < APPLY_PARALLEL; dim_offset++)
    {
#pragma HLS UNROLL
        int dim_in = dim_base + dim_offset;
        FM_TYPE message_dim = message[dim_in];
        FM_TYPE h_node_el = h_node_v[dim_in];

        // in preparation for next round of message passing
        message[dim_in] = 0;

        FM_TYPE activation;
        if (layer_num == 0)
        {
            activation = h_node_el;
        }
        else
        {
            WT_TYPE bn_sqrt_var_dim = bn_sqrt_var[layer_num - 1][dim_in];
            WT_TYPE bn_weight_dim = bn_weight[layer_num - 1][dim_in];
            WT_TYPE bn_mean_dim = bn_mean[layer_num - 1][dim_in];
            WT_TYPE bn_bias_dim = bn_bias[layer_num - 1][dim_in];
            WT_TYPE convs_root_emb_weight_dim = convs_root_emb_weight[layer_num - 1][dim_in];

            activation = message_dim + ap_fixed_relu<FM_TYPE>(h_node_el + convs_root_emb_weight_dim) / (degree + 1);
            activation = (activation - bn_mean_dim) / bn_sqrt_var_dim * bn_weight_dim + bn_bias_dim;
            activation = ap_fixed_relu<FM_TYPE>(activation);
        }

        for (int dim_out = 0; dim_out < EMB_DIM; dim_out++)
        {
#pragma HLS UNROLL
            FM_TYPE addend = activation * convs_weight[layer_num][dim_out][dim_in];
            FM_TYPE bias = convs_bias[layer_num][dim_out];
            accs[dim_out] = addend + ((dim_in == 0) ? bias : accs[dim_out]);
        }
    }
}

static void output(
    FM_TYPE h_node_v[EMB_DIM],
    FM_TYPE accs[EMB_DIM],
    hls::stream<ne_out_t>& embeddings,
    int dim_base
)
{
#pragma HLS INLINE
    ne_out_t output;
#pragma HLS AGGREGATE variable=output
    for (int dim_offset = 0; dim_offset < APPLY_PARALLEL; dim_offset++)
    {
#pragma HLS UNROLL
        int dim = dim_base + dim_offset;
        FM_TYPE result = accs[dim];
        h_node_v[dim] = result;
        output[dim_offset] = result;
    }
    embeddings << output;
}
