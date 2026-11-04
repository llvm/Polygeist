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
    WT_TYPE graph_pred_weights[NUM_TASK][EMB_DIM],
    WT_TYPE graph_pred_bias[NUM_TASK],
    FM_TYPE* result,
    int num_of_nodes
)
{
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    FM_TYPE h_graph[EMB_DIM];

    global_mean_pooling(embeddings, h_graph, num_of_nodes);
    linear<EMB_DIM, NUM_TASK, NUM_TASK, false>(
        h_graph,
        graph_pred_weights,
        graph_pred_bias,
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
