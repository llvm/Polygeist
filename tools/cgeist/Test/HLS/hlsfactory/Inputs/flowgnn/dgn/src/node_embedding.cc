#include "node_embedding.h"
#include "hls_math.h"

// #region Internal Function Declarations
static void accumulate(
    FM_TYPE message[2][EMB_DIM],
    FM_TYPE h_node_v[EMB_DIM],
    FM_TYPE accs[L_OUT],
    FM_TYPE h_node_buf[EMB_DIM],
    int degree,
    WT_TYPE eigw_sum,
    WT_TYPE eig_abssum,
    int layer_num,
    int dim_base
);
static void output(
    FM_TYPE h_node_v[EMB_DIM],
    FM_TYPE accs[L_OUT],
    FM_TYPE h_node_buf[EMB_DIM],
    hls::stream<ne_out_t>& embeddings,
    int dim_base
);
// #endregion

void node_embedding_multi_pe(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    FM_TYPE message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][2][EMB_DIM],
    int layer_num,
    int num_of_nodes
)
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=h_node cyclic factor=NODE_PARALLEL dim=1
#pragma HLS ARRAY_PARTITION variable=h_node cyclic factor=APPLY_PARALLEL dim=2

    FM_TYPE accs_ping[NODE_PARALLEL][L_OUT];
#pragma HLS ARRAY_PARTITION variable=accs_ping complete dim=1
#pragma HLS ARRAY_PARTITION variable=accs_ping complete dim=2
    FM_TYPE accs_pong[NODE_PARALLEL][L_OUT];
#pragma HLS ARRAY_PARTITION variable=accs_pong complete dim=1
#pragma HLS ARRAY_PARTITION variable=accs_pong complete dim=2
    FM_TYPE h_node_ping[NODE_PARALLEL][EMB_DIM];
#pragma HLS ARRAY_PARTITION variable=h_node_ping complete dim=1
#pragma HLS ARRAY_PARTITION variable=h_node_ping cyclic factor=APPLY_PARALLEL dim=2
    FM_TYPE h_node_pong[NODE_PARALLEL][EMB_DIM];
#pragma HLS ARRAY_PARTITION variable=h_node_pong complete dim=1
#pragma HLS ARRAY_PARTITION variable=h_node_pong cyclic factor=APPLY_PARALLEL dim=2

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
                            (i % 2 == 0) ? h_node_pong[v_offset] : h_node_ping[v_offset],
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
                            (i % 2 == 0) ? h_node_ping[v_offset] : h_node_pong[v_offset],
                            degree_table[v],
                            eigw_sums[v],
                            eig_abssums[v],
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
    FM_TYPE message[2][EMB_DIM],
    FM_TYPE h_node_v[EMB_DIM],
    FM_TYPE accs[L_OUT],
    FM_TYPE h_node_buf[EMB_DIM],
    int degree,
    WT_TYPE eigw_sum,
    WT_TYPE eig_abssum,
    int layer_num,
    int dim_base
)
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=layers_posttrans_fully_connected_0_linear_weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=layers_posttrans_fully_connected_0_linear_weight complete dim=3
#pragma HLS ARRAY_PARTITION variable=layers_posttrans_fully_connected_0_linear_weight cyclic factor=APPLY_PARALLEL dim=4
#pragma HLS ARRAY_PARTITION variable=layers_posttrans_fully_connected_0_linear_bias complete dim=2

    if (eig_abssum == 0.0)
    {
        eig_abssum = ap_fixed_epsilon<WT_TYPE>();
    }

    for (int dim_offset = 0; dim_offset < APPLY_PARALLEL; dim_offset++)
    {
#pragma HLS UNROLL
        int dim_in = dim_base + dim_offset;

        FM_TYPE message_1 = message[0][dim_in];
        FM_TYPE message_2 = message[1][dim_in];

        // clear message table in preparation for next round of message passing
        message[0][dim_in] = 0;
        message[1][dim_in] = 0;

        FM_TYPE h_node_el = h_node_v[dim_in];
        h_node_buf[dim_in] = h_node_el;

        FM_TYPE activation_1 = message_1 / degree;
        FM_TYPE activation_2 = hls::abs(FM_TYPE((message_2 - eigw_sum * h_node_el) / eig_abssum));

        for (int dim_out = 0; dim_out < L_OUT; dim_out++)
        {
#pragma HLS UNROLL
            FM_TYPE addend = (
                activation_1 * layers_posttrans_fully_connected_0_linear_weight[layer_num][dim_out][0][dim_in]
                + activation_2 * layers_posttrans_fully_connected_0_linear_weight[layer_num][dim_out][1][dim_in]
            );
            FM_TYPE bias = layers_posttrans_fully_connected_0_linear_bias[layer_num][dim_out];
            accs[dim_out] = addend + ((dim_in == 0) ? bias : accs[dim_out]);
        }
    }
}

static void output(
    FM_TYPE h_node_v[EMB_DIM],
    FM_TYPE accs[L_OUT],
    FM_TYPE h_node_buf[EMB_DIM],
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
        FM_TYPE acc = accs[dim];
        FM_TYPE relu_acc = (hls::signbit(acc)) ? FM_TYPE(0.0) : acc;
        FM_TYPE result = h_node_buf[dim] + relu_acc;
        h_node_v[dim] = result;
        output[dim_offset] = result;
    }
    embeddings << output;
}
