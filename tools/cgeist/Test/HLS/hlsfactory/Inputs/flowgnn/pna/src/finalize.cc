#include "finalize.h"
#include "linear.h"

using std::array;

// #region Internal Function Declarations
static void global_mean_pooling(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    FM_TYPE h_graph[EMB_DIM],
    int num_of_nodes
);
// #endregion

void finalize(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    WT_TYPE graph_mlp_1_weights[GRAPH_MLP_1_OUT][EMB_DIM],
    WT_TYPE graph_mlp_1_bias[GRAPH_MLP_1_OUT],
    WT_TYPE graph_mlp_2_weights[GRAPH_MLP_2_OUT][GRAPH_MLP_1_OUT],
    WT_TYPE graph_mlp_2_bias[GRAPH_MLP_2_OUT],
    WT_TYPE graph_mlp_3_weights[NUM_TASK][GRAPH_MLP_2_OUT],
    WT_TYPE graph_mlp_3_bias[NUM_TASK],
    FM_TYPE* result,
    int num_of_nodes
)
{
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    FM_TYPE h_graph[EMB_DIM];
    hls::stream<mlp_xfer_t> graph_mlp_1_out("graph_mlp_1_out");
#pragma HLS STREAM variable=graph_mlp_1_out depth=ceildiv(GRAPH_MLP_1_OUT, MLP_PARALLEL)
    FM_TYPE graph_mlp_2_out[GRAPH_MLP_2_OUT];

    global_mean_pooling(embeddings, h_graph, num_of_nodes);
    linear_output_stationary<EMB_DIM, GRAPH_MLP_1_OUT, MLP_PARALLEL>(
        h_graph,
        graph_mlp_1_weights,
        graph_mlp_1_bias,
        graph_mlp_1_out
    );
    linear_input_stationary<GRAPH_MLP_1_OUT, GRAPH_MLP_2_OUT, MLP_PARALLEL>(
        graph_mlp_1_out,
        graph_mlp_2_weights,
        graph_mlp_2_bias,
        graph_mlp_2_out
    );
    linear<GRAPH_MLP_2_OUT, NUM_TASK, NUM_TASK, false>(
        graph_mlp_2_out,
        graph_mlp_3_weights,
        graph_mlp_3_bias,
        result
    );
}

static void global_mean_pooling(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    FM_TYPE h_graph[EMB_DIM],
    int num_of_nodes
)
{
#pragma HLS INLINE off

    FM_TYPE sums[EMB_DIM];
#pragma HLS ARRAY_PARTITION variable=sums cyclic factor=APPLY_PARALLEL dim=1

    int num_iters = ceildiv(num_of_nodes, NODE_PARALLEL) - 1;
    int tail_nodes = ((num_of_nodes - 1) % NODE_PARALLEL) + 1;

    global_mean_pooling_main: for (int i = 0; i < num_iters; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min=(ceildiv(ANALYSIS_MIN_NODES, NODE_PARALLEL) - 1) max=(ceildiv(ANALYSIS_MAX_NODES, NODE_PARALLEL) - 1) avg=(ceildiv(ANALYSIS_AVG_NODES, NODE_PARALLEL) - 1)
        for (int dim_base = 0; dim_base < EMB_DIM; dim_base += APPLY_PARALLEL)
        {
#pragma HLS PIPELINE II=1

            ne_out_t embeddings_slice[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=embeddings_slice complete dim=1

            for (int nd_offset = 0; nd_offset < NODE_PARALLEL; nd_offset++)
            {
#pragma HLS UNROLL
                embeddings[nd_offset] >> embeddings_slice[nd_offset];
            }

            for (int dim_offset = 0; dim_offset < APPLY_PARALLEL; dim_offset++)
            {
#pragma HLS UNROLL
                int dim = dim_base + dim_offset;
                FM_TYPE h_graph_el = 0;

                for (int nd_offset = 0; nd_offset < NODE_PARALLEL; nd_offset++)
                {
#pragma HLS UNROLL
                    h_graph_el += embeddings_slice[nd_offset][dim_offset];
                }

                if (i != 0) h_graph_el += sums[dim];
                sums[dim] = h_graph_el;
            }
        }
    }

    global_mean_pooling_tail: for (int dim_base = 0; dim_base < EMB_DIM; dim_base += APPLY_PARALLEL)
    {
#pragma HLS PIPELINE II=1

        ne_out_t embeddings_slice[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=embeddings_slice complete dim=1

        for (int nd_offset = 0; nd_offset < NODE_PARALLEL; nd_offset++)
        {
#pragma HLS UNROLL
            if (nd_offset == tail_nodes) break;
            embeddings[nd_offset] >> embeddings_slice[nd_offset];
        }

        for (int dim_offset = 0; dim_offset < APPLY_PARALLEL; dim_offset++)
        {
#pragma HLS UNROLL
            int dim = dim_base + dim_offset;
            FM_TYPE h_graph_el = 0;

            for (int nd_offset = 0; nd_offset < NODE_PARALLEL; nd_offset++)
            {
#pragma HLS UNROLL
                if (nd_offset == tail_nodes) break;
                h_graph_el += embeddings_slice[nd_offset][dim_offset];
            }

            if (num_iters != 0) h_graph_el += sums[dim];
            h_graph[dim] = h_graph_el / num_of_nodes;
        }
    }
}
