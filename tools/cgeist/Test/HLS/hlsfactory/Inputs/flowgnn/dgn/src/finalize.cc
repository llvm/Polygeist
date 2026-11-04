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
    WT_TYPE MLP_layer_FC_layers_0_weight[50][100],
    WT_TYPE MLP_layer_FC_layers_0_bias[50],
    WT_TYPE MLP_layer_FC_layers_1_weight[25][50],
    WT_TYPE MLP_layer_FC_layers_1_bias[25],
    WT_TYPE MLP_layer_FC_layers_2_weight[1][25],
    WT_TYPE MLP_layer_FC_layers_2_bias[1],
    FM_TYPE* result,
    int num_of_nodes
)
{
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    FM_TYPE h_graph[EMB_DIM];
    hls::stream<mlp_xfer_t> mlp_0_out("mlp_0_out");
#pragma HLS STREAM variable=mlp_0_out depth=50
    FM_TYPE mlp_1_out[25];

    global_mean_pooling(embeddings, h_graph, num_of_nodes);
    linear_output_stationary<100, 50, 2>(
        h_graph,
        MLP_layer_FC_layers_0_weight,
        MLP_layer_FC_layers_0_bias,
        mlp_0_out
    );
    linear_input_stationary<50, 25, 2>(
        mlp_0_out,
        MLP_layer_FC_layers_1_weight,
        MLP_layer_FC_layers_1_bias,
        mlp_1_out
    );
    linear<25, 1, 1, false>(
        mlp_1_out,
        MLP_layer_FC_layers_2_weight,
        MLP_layer_FC_layers_2_bias,
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

    int num_iters = ceildiv(num_of_nodes, NODE_PARALLEL);
    for (int i = 0, nd_base = 0; i < num_iters; i++, nd_base += NODE_PARALLEL)
    {
#pragma HLS LOOP_TRIPCOUNT min=ceildiv(ANALYSIS_MIN_NODES, NODE_PARALLEL) max=ceildiv(ANALYSIS_MAX_NODES, NODE_PARALLEL) avg=ceildiv(ANALYSIS_AVG_NODES, NODE_PARALLEL)
        for (int dim_base = 0; dim_base < EMB_DIM; dim_base += APPLY_PARALLEL)
        {
#pragma HLS PIPELINE II=1

            ne_out_t embeddings_slice[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=embeddings_slice complete dim=1

            for (int nd_offset = 0; nd_offset < NODE_PARALLEL; nd_offset++)
            {
#pragma HLS UNROLL
                int nd = nd_base + nd_offset;
                if (nd == num_of_nodes) break;
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
                    int nd = nd_base + nd_offset;
                    if (nd == num_of_nodes) break;
                    h_graph_el += embeddings_slice[nd_offset][dim_offset];
                }

                if (nd_base != 0) h_graph_el += sums[dim];
                sums[dim] = h_graph_el;
                h_graph[dim] = h_graph_el / num_of_nodes;
            }
        }
    }
}
