#include "finalize.h"
#include "linear.h"

using std::array;

// #region Internal Function Declarations
static void post_mp_then_global_mean_pooling(
    FM_TYPE h_node[MAX_NODE][EMB_DIM],
    FM_TYPE message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    hls::stream<ne_out_t>& h_graph,
    int num_of_nodes
);
// #endregion

void finalize(
    FM_TYPE h_node[MAX_NODE][EMB_DIM],
    FM_TYPE message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    WT_TYPE graph_pred_weights[NUM_TASK][EMB_DIM],
    WT_TYPE graph_pred_bias[NUM_TASK],
    FM_TYPE* result,
    int num_of_nodes
)
{
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<ne_out_t> h_graph;
#pragma HLS STREAM variable=h_graph depth=ceildiv(EMB_DIM, APPLY_PARALLEL)

    post_mp_then_global_mean_pooling(h_node, message, h_graph, num_of_nodes);
    linear_input_stationary<EMB_DIM, NUM_TASK, APPLY_PARALLEL, false>(
        h_graph,
        graph_pred_weights,
        graph_pred_bias,
        result
    );
}

static void post_mp_then_global_mean_pooling(
    FM_TYPE h_node[MAX_NODE][EMB_DIM],
    FM_TYPE message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    hls::stream<ne_out_t>& h_graph,
    int num_of_nodes
)
{
#pragma HLS INLINE off

    FM_TYPE sums[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=sums complete dim=1

    int num_iters = ceildiv(num_of_nodes, NODE_PARALLEL) * ceildiv(EMB_DIM, APPLY_PARALLEL);
    for (int i = 0, next_nd_base = 0, next_dim_base = 0; i < num_iters; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min=(ceildiv(ANALYSIS_MIN_NODES, NODE_PARALLEL) * ceildiv(EMB_DIM, APPLY_PARALLEL)) max=(ceildiv(ANALYSIS_MAX_NODES, NODE_PARALLEL) * ceildiv(EMB_DIM, APPLY_PARALLEL)) avg=(ceildiv(ANALYSIS_AVG_NODES, NODE_PARALLEL) * ceildiv(EMB_DIM, APPLY_PARALLEL))
#pragma HLS PIPELINE II=1

        int nd_base = next_nd_base;
        int dim_base = next_dim_base;

        next_nd_base = nd_base + NODE_PARALLEL;
        if (next_nd_base >= num_of_nodes)
        {
            next_nd_base = 0;
            next_dim_base = dim_base + APPLY_PARALLEL;
        }

        if (nd_base == 0)
        {
            for (int dim_offset = 0; dim_offset < APPLY_PARALLEL; dim_offset++)
            {
#pragma HLS UNROLL
                sums[dim_offset] = 0;
            }
        }

        for (int dim_offset = 0; dim_offset < APPLY_PARALLEL; dim_offset++)
        {
#pragma HLS UNROLL
            int dim = dim_base + dim_offset;
            if (dim == EMB_DIM) break;

            WT_TYPE convs_root_emb_weight_dim = convs_root_emb_weight[NUM_LAYERS - 1][dim];
            WT_TYPE bn_sqrt_var_dim = bn_sqrt_var[NUM_LAYERS - 1][dim];
            WT_TYPE bn_weight_dim = bn_weight[NUM_LAYERS - 1][dim];
            WT_TYPE bn_mean_dim = bn_mean[NUM_LAYERS - 1][dim];
            WT_TYPE bn_bias_dim = bn_bias[NUM_LAYERS - 1][dim];

            for (int nd_offset = 0; nd_offset < NODE_PARALLEL; nd_offset++)
            {
#pragma HLS UNROLL
                int nd = nd_base + nd_offset;
                if (nd == num_of_nodes) break;

                FM_TYPE activation = message[nd % EDGE_PARALLEL][nd / EDGE_PARALLEL][dim];
                activation += ap_fixed_relu<FM_TYPE>(h_node[nd][dim] + convs_root_emb_weight_dim) / (degree_table[nd] + 1);
                activation = (activation - bn_mean_dim) / bn_sqrt_var_dim * bn_weight_dim + bn_bias_dim;
                sums[dim_offset] += activation;
            }
        }

        if (next_nd_base == 0)
        {
            ne_out_t h_graph_vals;
#pragma HLS AGGREGATE variable=h_graph_vals

            for (int dim_offset = 0; dim_offset < APPLY_PARALLEL; dim_offset++)
            {
#pragma HLS UNROLL
                h_graph_vals[dim_offset] = sums[dim_offset] / num_of_nodes;
            }

            h_graph << h_graph_vals;
        }
    }
}
