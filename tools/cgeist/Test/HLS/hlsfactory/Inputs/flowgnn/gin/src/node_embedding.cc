#include "node_embedding.h"

// #region Internal Function Declarations
static void accumulate(
    FM_TYPE message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_TYPE accs[NODE_PARALLEL][MLP_1_OUT],
    WT_TYPE eps,
    int layer_num,
    int v_base,
    int dim_base,
    int num_of_nodes
);
static void output(
    FM_TYPE accs[NODE_PARALLEL][MLP_1_OUT],
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    int layer_num,
    int v_base,
    int dim_base,
    int num_of_nodes
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
#pragma HLS ARRAY_PARTITION variable=h_node cyclic factor=NODE_PARALLEL dim=1
#pragma HLS ARRAY_PARTITION variable=h_node cyclic factor=APPLY_PARALLEL dim=2

    FM_TYPE accs_ping[NODE_PARALLEL][MLP_1_OUT];
#pragma HLS ARRAY_PARTITION variable=accs_ping complete dim=1
#pragma HLS ARRAY_PARTITION variable=accs_ping complete dim=2
    FM_TYPE accs_pong[NODE_PARALLEL][MLP_1_OUT];
#pragma HLS ARRAY_PARTITION variable=accs_pong complete dim=1
#pragma HLS ARRAY_PARTITION variable=accs_pong complete dim=2

    WT_TYPE eps = node_mlp_eps[layer_num];
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
                output(
                    (i % 2 == 0) ? accs_pong : accs_ping,
                    embeddings,
                    layer_num,
                    out_v_base,
                    dim_base,
                    num_of_nodes
                );
            }

            if (i != num_iters - 1)
            {
                accumulate(
                    message,
                    (i % 2 == 0) ? accs_ping : accs_pong,
                    eps,
                    layer_num,
                    acc_v_base,
                    dim_base,
                    num_of_nodes
                );
            }
        }
    }
}

static void accumulate(
    FM_TYPE message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_TYPE accs[NODE_PARALLEL][MLP_1_OUT],
    WT_TYPE eps,
    int layer_num,
    int v_base,
    int dim_base,
    int num_of_nodes
)
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=node_mlp_1_weights complete dim=2
#pragma HLS ARRAY_PARTITION variable=node_mlp_1_weights cyclic factor=APPLY_PARALLEL dim=3
#pragma HLS ARRAY_PARTITION variable=node_mlp_1_bias complete dim=2

    for (int dim_offset = 0; dim_offset < APPLY_PARALLEL; dim_offset++)
    {
#pragma HLS UNROLL
        int dim_in = dim_base + dim_offset;
        FM_TYPE h_node_els[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=h_node_els complete dim=1
        FM_TYPE activations[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=activations complete dim=1

        for (int v_offset = 0; v_offset < NODE_PARALLEL; v_offset++)
        {
            int v = v_base + v_offset;
            h_node_els[v_offset] = 0;
            activations[v_offset] = 0;

            if (v < num_of_nodes)
            {
                FM_TYPE& message_dim = message[v % EDGE_PARALLEL][v / EDGE_PARALLEL][dim_in];
                h_node_els[v_offset] = h_node[v][dim_in];
                activations[v_offset] = message_dim + (1 + eps) * h_node_els[v_offset];

                // in preparation for next round of message passing
                message_dim = 0;
            }
        }

        for (int dim_out = 0; dim_out < MLP_1_OUT; dim_out++)
        {
#pragma HLS UNROLL
            WT_TYPE weight = node_mlp_1_weights[layer_num][dim_out][dim_in];
            FM_TYPE bias = node_mlp_1_bias[layer_num][dim_out];
            for (int v_offset = 0; v_offset < NODE_PARALLEL; v_offset++)
            {
#pragma HLS UNROLL
                FM_TYPE addend = activations[v_offset] * weight;
                accs[v_offset][dim_out] = addend + ((dim_in == 0) ? bias : accs[v_offset][dim_out]);
            }
        }
    }
}

static void output(
    FM_TYPE accs[NODE_PARALLEL][MLP_1_OUT],
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    int layer_num,
    int v_base,
    int dim_base,
    int num_of_nodes
)
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=node_mlp_2_weights cyclic factor=APPLY_PARALLEL dim=2
#pragma HLS ARRAY_PARTITION variable=node_mlp_2_weights complete dim=3
#pragma HLS ARRAY_PARTITION variable=node_mlp_2_bias cyclic factor=APPLY_PARALLEL dim=2

    ne_out_t outputs[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=outputs complete dim=1
#pragma HLS AGGREGATE variable=outputs

    for (int dim_offset = 0; dim_offset < APPLY_PARALLEL; dim_offset++)
    {
#pragma HLS UNROLL
        int dim = dim_base + dim_offset;

        FM_TYPE results[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=results complete dim=1

        FM_TYPE bias = node_mlp_2_bias[layer_num][dim];
        for (int v_offset = 0; v_offset < NODE_PARALLEL; v_offset++)
        {
#pragma HLS UNROLL
            results[v_offset] = bias;
        }

        for (int dim_in = 0; dim_in < MLP_1_OUT; dim_in++)
        {
#pragma HLS UNROLL
            WT_TYPE weight = node_mlp_2_weights[layer_num][dim][dim_in];
            for (int v_offset = 0; v_offset < NODE_PARALLEL; v_offset++)
            {
#pragma HLS UNROLL
                FM_TYPE activation = accs[v_offset][dim_in];
                results[v_offset] += ap_fixed_relu(activation) * weight;
            }
        }

        for (int v_offset = 0; v_offset < NODE_PARALLEL; v_offset++)
        {
#pragma HLS UNROLL
            int v = v_base + v_offset;
            FM_TYPE result = results[v_offset];
            if (layer_num != NUM_LAYERS - 1) result = ap_fixed_relu(result);
            outputs[v_offset][dim_offset] = result;
            if (v < num_of_nodes) h_node[v][dim] = result;
        }
    }

    for (int v_offset = 0; v_offset < NODE_PARALLEL; v_offset++)
    {
#pragma HLS UNROLL
        int v = v_base + v_offset;
        if (v < num_of_nodes) embeddings[v_offset] << outputs[v_offset];
    }
}
