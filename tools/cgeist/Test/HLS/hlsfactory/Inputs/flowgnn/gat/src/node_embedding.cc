#include "node_embedding.h"

// #region Internal Function Declarations
static void accumulate(
    hls::stream<ne_in_t> messages[NODE_PARALLEL],
    FM_VEC accs[NODE_PARALLEL][EMB_DIM],
    FM_VEC out_nodes_features_skip_concat_bias[MAX_NODE][EMB_DIM],
    FM_VEC next_out_nodes_features_skip_concat_bias[MAX_NODE][EMB_DIM],
    int layer_num,
    int v_base,
    int dim_base,
    int num_of_nodes
);
static void output(
    FM_VEC accs[NODE_PARALLEL][EMB_DIM],
    FM_VEC scores_source_accs[NODE_PARALLEL],
    FM_VEC scores_target_accs[NODE_PARALLEL],
    FM_VEC h_node[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_VEC scores_source[EDGE_PARALLEL][MAX_NODE],
    FM_VEC scores_target[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)],
    int layer_num,
    int v_base,
    int dim_base,
    int num_of_nodes
);
// #endregion

void node_embedding_multi_pe(
    hls::stream<ne_in_t> messages[NODE_PARALLEL],
    FM_VEC h_node[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_VEC out_nodes_features_skip_concat_bias[MAX_NODE][EMB_DIM],
    FM_VEC next_out_nodes_features_skip_concat_bias[MAX_NODE][EMB_DIM],
    FM_VEC scores_source[EDGE_PARALLEL][MAX_NODE],
    FM_VEC scores_target[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)],
    int layer_num,
    int num_of_nodes
)
{
#pragma HLS INLINE off

    FM_VEC accs_ping[NODE_PARALLEL][EMB_DIM];
#pragma HLS ARRAY_PARTITION variable=accs_ping complete dim=0
    FM_VEC accs_pong[NODE_PARALLEL][EMB_DIM];
#pragma HLS ARRAY_PARTITION variable=accs_pong complete dim=0
    FM_VEC scores_source_accs[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=scores_source_accs complete dim=0
    FM_VEC scores_target_accs[NODE_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=scores_target_accs complete dim=0

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
#pragma HLS DEPENDENCE variable=h_node inter false
#pragma HLS DEPENDENCE variable=scores_source inter false
#pragma HLS DEPENDENCE variable=scores_target inter false

            if (i != 0)
            {
                output(
                    (i % 2 == 0) ? accs_pong : accs_ping,
                    scores_source_accs,
                    scores_target_accs,
                    h_node,
                    scores_source,
                    scores_target,
                    layer_num,
                    out_v_base,
                    dim_base,
                    num_of_nodes
                );
            }

            if (i != num_iters - 1)
            {
                accumulate(
                    messages,
                    (i % 2 == 0) ? accs_ping : accs_pong,
                    out_nodes_features_skip_concat_bias,
                    next_out_nodes_features_skip_concat_bias,
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
    hls::stream<ne_in_t> messages[NODE_PARALLEL],
    FM_VEC accs[NODE_PARALLEL][EMB_DIM],
    FM_VEC out_nodes_features_skip_concat_bias[MAX_NODE][EMB_DIM],
    FM_VEC next_out_nodes_features_skip_concat_bias[MAX_NODE][EMB_DIM],
    int layer_num,
    int v_base,
    int dim_base,
    int num_of_nodes
)
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=out_nodes_features_skip_concat_bias cyclic factor=NODE_PARALLEL dim=1
#pragma HLS ARRAY_PARTITION variable=out_nodes_features_skip_concat_bias complete dim=2
#pragma HLS ARRAY_PARTITION variable=next_out_nodes_features_skip_concat_bias cyclic factor=NODE_PARALLEL dim=1
#pragma HLS ARRAY_PARTITION variable=next_out_nodes_features_skip_concat_bias complete dim=2
#pragma HLS ARRAY_PARTITION variable=skip_proj_weights complete dim=2
#pragma HLS ARRAY_PARTITION variable=skip_proj_weights cyclic factor=APPLY_PARALLEL dim=3
#pragma HLS ARRAY_PARTITION variable=skip_proj_weights complete dim=4
#pragma HLS ARRAY_PARTITION variable=skip_proj_weights complete dim=5
#pragma HLS ARRAY_PARTITION variable=linear_proj_weights complete dim=2
#pragma HLS ARRAY_PARTITION variable=linear_proj_weights complete dim=3
#pragma HLS ARRAY_PARTITION variable=linear_proj_weights complete dim=4
#pragma HLS ARRAY_PARTITION variable=linear_proj_weights cyclic factor=APPLY_PARALLEL dim=5

    WT_TYPE curr_skip_proj_weights[NUM_HEADS][APPLY_PARALLEL][NUM_HEADS][EMB_DIM];
#pragma HLS ARRAY_PARTITION variable=curr_skip_proj_weights complete dim=0
    WT_TYPE curr_linear_proj_weights[NUM_HEADS][EMB_DIM][NUM_HEADS][APPLY_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=curr_linear_proj_weights complete dim=0

    for (int dim_offset = 0; dim_offset < APPLY_PARALLEL; dim_offset++)
    {
        int dim = dim_base + dim_offset;
        for (int head_out = 0; head_out < NUM_HEADS; head_out++)
        {
            for (int other_dim = 0; other_dim < EMB_DIM; other_dim++)
            {
                for (int head_in = 0; head_in < NUM_HEADS; head_in++)
                {
                    curr_skip_proj_weights[head_out][dim_offset][head_in][other_dim] = skip_proj_weights[layer_num][head_out][dim][head_in][other_dim];
                    curr_linear_proj_weights[head_out][other_dim][head_in][dim_offset] = linear_proj_weights[layer_num + 1][head_out][other_dim][head_in][dim];
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
            messages[v_offset] >> message;

            for (int dim_out_offset = 0; dim_out_offset < APPLY_PARALLEL; dim_out_offset++)
            {
                int dim_out = dim_base + dim_out_offset;

                // prepare_out_nodes_features()
                FM_VEC next_out_node_feature_skip_concat_bias = message[dim_out_offset];
                for (int dim_in = 0; dim_in < EMB_DIM; dim_in++)
                {
                    FM_VEC activation = out_nodes_features_skip_concat_bias[v][dim_in];
                    for (int head_out = 0; head_out < NUM_HEADS; head_out++)
                    {
                        for (int head_in = 0; head_in < NUM_HEADS; head_in++)
                        {
                            WT_TYPE weight = curr_skip_proj_weights[head_out][dim_out_offset][head_in][dim_in];
                            next_out_node_feature_skip_concat_bias[head_out] += activation[head_in] * weight;
                        }
                    }
                }

                // compute_activation()
                for (int head_out = 0; head_out < NUM_HEADS; head_out++)
                {
                    if (next_out_node_feature_skip_concat_bias[head_out] <= 0)
                    {
                        next_out_node_feature_skip_concat_bias[head_out] = hls::exp(next_out_node_feature_skip_concat_bias[head_out]) - FM_TYPE(1);
                    }
                }
                next_out_nodes_features_skip_concat_bias[v][dim_out] = next_out_node_feature_skip_concat_bias;

                // compute_nodes_features_proj()
                for (int proj_dim_out = 0; proj_dim_out < EMB_DIM; proj_dim_out++)
                {
                    FM_VEC acc = (dim_out != 0) ? accs[v_offset][proj_dim_out] : FM_VEC(0);
                    for (int head_in = 0; head_in < NUM_HEADS; head_in++)
                    {
                        WT_VEC weight;
                        for (int head_out = 0; head_out < NUM_HEADS; head_out++)
                        {
                            weight[head_out] = curr_linear_proj_weights[head_out][proj_dim_out][head_in][dim_out_offset];
                        }
                        acc += next_out_node_feature_skip_concat_bias[head_in] * weight;
                    }
                    accs[v_offset][proj_dim_out] = acc;
                }
            }
        }
    }
}

static void output(
    FM_VEC accs[NODE_PARALLEL][EMB_DIM],
    FM_VEC scores_source_accs[NODE_PARALLEL],
    FM_VEC scores_target_accs[NODE_PARALLEL],
    FM_VEC h_node[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_VEC scores_source[EDGE_PARALLEL][MAX_NODE],
    FM_VEC scores_target[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)],
    int layer_num,
    int v_base,
    int dim_base,
    int num_of_nodes
)
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=scoring_fn_source complete dim=2
#pragma HLS ARRAY_PARTITION variable=scoring_fn_source cyclic factor=APPLY_PARALLEL dim=3
#pragma HLS ARRAY_PARTITION variable=scoring_fn_target complete dim=2
#pragma HLS ARRAY_PARTITION variable=scoring_fn_target cyclic factor=APPLY_PARALLEL dim=3

    WT_VEC scoring_fn_source_weights[APPLY_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=scoring_fn_source_weights complete dim=0
    WT_VEC scoring_fn_target_weights[APPLY_PARALLEL];
#pragma HLS ARRAY_PARTITION variable=scoring_fn_target_weights complete dim=0

    for (int dim_offset = 0; dim_offset < APPLY_PARALLEL; dim_offset++)
    {
        int dim = dim_base + dim_offset;
        for (int head = 0; head < NUM_HEADS; head++)
        {
            scoring_fn_source_weights[dim_offset][head] = scoring_fn_source[layer_num + 1][head][dim];
            scoring_fn_target_weights[dim_offset][head] = scoring_fn_target[layer_num + 1][head][dim];
        }
    }

    for (int v_offset = 0; v_offset < NODE_PARALLEL; v_offset++)
    {
        int v = v_base + v_offset;
        if (v < num_of_nodes)
        {
            FM_VEC scores_source_acc = FM_TYPE(0);
            FM_VEC scores_target_acc = FM_TYPE(0);

            for (int dim_offset = 0; dim_offset < APPLY_PARALLEL; dim_offset++)
            {
                int dim = dim_base + dim_offset;
                FM_VEC result = accs[v_offset][dim];
                h_node[v % EDGE_PARALLEL][v / EDGE_PARALLEL][dim] = result;
                scores_source_acc += result * scoring_fn_source_weights[dim_offset];
                scores_target_acc += result * scoring_fn_target_weights[dim_offset];
            }

            if (dim_base != 0)
            {
                scores_source_acc += scores_source_accs[v_offset];
                scores_target_acc += scores_target_accs[v_offset];
            }

            scores_source_accs[v_offset] = scores_source_acc;
            scores_target_accs[v_offset] = scores_target_acc;

            if (dim_base == ((EMB_DIM - 1) / APPLY_PARALLEL) * APPLY_PARALLEL)
            {
                for (int pe_id = 0; pe_id < EDGE_PARALLEL; pe_id++)
                {
                    scores_source[pe_id][v] = scores_source_acc;
                }
                scores_target[v % EDGE_PARALLEL][v / EDGE_PARALLEL] = scores_target_acc;
            }
        }
    }
}
