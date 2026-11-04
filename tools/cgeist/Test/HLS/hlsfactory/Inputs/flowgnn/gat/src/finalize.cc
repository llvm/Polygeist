#include "finalize.h"
#include "linear.h"

using std::array;

// #region Internal Function Declarations
static void make_embeddings(
    hls::stream<ne_in_t> messages[NODE_PARALLEL],
    hls::stream<hls::vector<FM_TYPE, APPLY_PARALLEL>> embeddings[NODE_PARALLEL],
    FM_VEC out_nodes_features_skip_concat_bias[MAX_NODE][EMB_DIM],
    int num_of_nodes
);
static void global_mean_pooling(
    hls::stream<hls::vector<FM_TYPE, APPLY_PARALLEL>> embeddings[NODE_PARALLEL],
    FM_TYPE h_graph[EMB_DIM],
    int num_of_nodes
);
// #endregion

void finalize(
    hls::stream<ne_in_t> messages[NODE_PARALLEL],
    FM_VEC out_nodes_features_skip_concat_bias[MAX_NODE][EMB_DIM],
    WT_TYPE graph_pred_weights[NUM_TASK][EMB_DIM],
    WT_TYPE graph_pred_bias[NUM_TASK],
    FM_TYPE* result,
    int num_of_nodes
)
{
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<hls::vector<FM_TYPE, APPLY_PARALLEL>> embeddings[NODE_PARALLEL];
#pragma HLS STREAM variable=embeddings depth=(4 * ceildiv(EMB_DIM, APPLY_PARALLEL))
    FM_TYPE h_graph[EMB_DIM];

    make_embeddings(messages, embeddings, out_nodes_features_skip_concat_bias, num_of_nodes);
    global_mean_pooling(embeddings, h_graph, num_of_nodes);
    linear<EMB_DIM, NUM_TASK, NUM_TASK, false>(
        h_graph,
        graph_pred_weights,
        graph_pred_bias,
        result
    );
}

static void make_embeddings(
    hls::stream<ne_in_t> messages[NODE_PARALLEL],
    hls::stream<hls::vector<FM_TYPE, APPLY_PARALLEL>> embeddings[NODE_PARALLEL],
    FM_VEC out_nodes_features_skip_concat_bias[MAX_NODE][EMB_DIM],
    int num_of_nodes
)
{
#pragma HLS INLINE off

    WT_TYPE curr_skip_proj_weights[NUM_HEADS][APPLY_PARALLEL][NUM_HEADS][EMB_DIM];
#pragma HLS ARRAY_PARTITION variable=curr_skip_proj_weights complete dim=0

    int num_iters = ceildiv(num_of_nodes, NODE_PARALLEL);
    for (int i = 0, v_base = 0; i < num_iters; i++, v_base += NODE_PARALLEL)
    {
#pragma HLS LOOP_TRIPCOUNT min=ceildiv(ANALYSIS_MIN_NODES, NODE_PARALLEL) max=ceildiv(ANALYSIS_MAX_NODES, NODE_PARALLEL) avg=ceildiv(ANALYSIS_AVG_NODES, NODE_PARALLEL)
        for (int dim_base = 0; dim_base < EMB_DIM; dim_base += APPLY_PARALLEL)
        {
#pragma HLS PIPELINE II=1
            for (int dim_offset = 0; dim_offset < APPLY_PARALLEL; dim_offset++)
            {
                int dim = dim_base + dim_offset;
                for (int head_out = 0; head_out < NUM_HEADS; head_out++)
                {
                    for (int other_dim = 0; other_dim < EMB_DIM; other_dim++)
                    {
                        for (int head_in = 0; head_in < NUM_HEADS; head_in++)
                        {
                            curr_skip_proj_weights[head_out][dim_offset][head_in][other_dim] = skip_proj_weights[NUM_LAYERS - 1][head_out][dim][head_in][other_dim];
                        }
                    }
                }
            }

            for (int v_offset = 0; v_offset < NODE_PARALLEL; v_offset++)
            {
                int v = v_base + v_offset;
                if (v < num_of_nodes)
                {
                    ne_in_t message;
                    hls::vector<FM_TYPE, APPLY_PARALLEL> embedding;
                    messages[v_offset] >> message;

                    // prepare_out_nodes_features() & compute_not_concat()
                    for (int dim_out_offset = 0; dim_out_offset < APPLY_PARALLEL; dim_out_offset++)
                    {
                        FM_TYPE out_node_features = 0;
                        for (int head = 0; head < NUM_HEADS; head++)
                        {
                            out_node_features += message[dim_out_offset][head];
                        }
                        for (int dim_in = 0; dim_in < EMB_DIM; dim_in++)
                        {
                            FM_VEC activation = out_nodes_features_skip_concat_bias[v][dim_in];
                            for (int head_out = 0; head_out < NUM_HEADS; head_out++)
                            {
                                for (int head_in = 0; head_in < NUM_HEADS; head_in++)
                                {
                                    WT_TYPE weight = curr_skip_proj_weights[head_out][dim_out_offset][head_in][dim_in];
                                    out_node_features += activation[head_in] * weight;
                                }
                            }
                        }
                        embedding[dim_out_offset] = out_node_features / NUM_HEADS;
                    }

                    embeddings[v_offset] << embedding;
                }
            }
        }
    }
}

static void global_mean_pooling(
    hls::stream<hls::vector<FM_TYPE, APPLY_PARALLEL>> embeddings[NODE_PARALLEL],
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

            hls::vector<FM_TYPE, APPLY_PARALLEL> embeddings_slice[NODE_PARALLEL];
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

        hls::vector<FM_TYPE, APPLY_PARALLEL> embeddings_slice[NODE_PARALLEL];
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
