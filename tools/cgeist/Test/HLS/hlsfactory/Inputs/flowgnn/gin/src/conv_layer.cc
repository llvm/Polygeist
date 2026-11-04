#include "conv_layer.h"
#include "node_embedding.h"
#include "message_passing.h"
#include "load_inputs.h"
#include "finalize.h"
#include "hls_stream.h"

// #region Internal Function Declarations
void check_node_embedding(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    FM_TYPE message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    node_feature_t* node_feature_in,
    WT_TYPE node_embedding_weight_in[ND_FEATURE_TOTAL][EMB_DIM],
    int i,
    int num_of_nodes
);
void check_message_passing(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    FM_TYPE message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_TYPE* result,
    int layer_num,
    int num_of_nodes
);
void message_passing_all_pes(
    hls::stream<ne_out_t> ne_out[NODE_PARALLEL],
    FM_TYPE message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    int layer_num,
    int num_of_nodes
);
void ne_to_mp_adapter(
    hls::stream<ne_out_t> ne_out[NODE_PARALLEL],
    hls::stream<mp_in_t> mp_in[EDGE_PARALLEL][NODE_PARALLEL],
    int num_of_nodes
);
// #endregion

void compute_CONV_layer(
    int layer_num,
    FM_TYPE message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_TYPE next_message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    node_feature_t* node_feature_in,
    WT_TYPE node_embedding_weight_in[ND_FEATURE_TOTAL][EMB_DIM],
    FM_TYPE* result,
    int num_of_nodes
)
{
#pragma HLS INLINE off
#pragma HLS DATAFLOW

#pragma HLS ARRAY_PARTITION variable=message cyclic factor=SCATTER_PARALLEL dim=3
#pragma HLS ARRAY_PARTITION variable=next_message cyclic factor=SCATTER_PARALLEL dim=3

    hls::stream<ne_out_t> embeddings[NODE_PARALLEL];
#pragma HLS STREAM variable=embeddings depth=(4 * ceildiv(EMB_DIM, APPLY_PARALLEL))

    check_node_embedding(embeddings, message, node_feature_in, node_embedding_weight_in, layer_num, num_of_nodes);
    check_message_passing(embeddings, next_message, result, layer_num, num_of_nodes);
}

void check_node_embedding(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    FM_TYPE message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    node_feature_t* node_feature_in,
    WT_TYPE node_embedding_weight_in[ND_FEATURE_TOTAL][EMB_DIM],
    int i,
    int num_of_nodes
)
{
#pragma HLS INLINE off

    if (i == 0)
        load_input_node_embeddings(embeddings, node_feature_in, node_embedding_weight_in, message, num_of_nodes);
    else
        node_embedding_multi_pe(embeddings, message, i - 1, num_of_nodes);
}

void check_message_passing(
    hls::stream<ne_out_t> embeddings[NODE_PARALLEL],
    FM_TYPE message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_TYPE* result,
    int layer_num,
    int num_of_nodes
)
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=message complete dim=1

    if (layer_num == NUM_LAYERS)
        finalize(
            embeddings,
            graph_pred_weights,
            graph_pred_bias,
            result,
            num_of_nodes
        );
    else
        message_passing_all_pes(embeddings, message, layer_num, num_of_nodes);
}

void message_passing_all_pes(
    hls::stream<ne_out_t> ne_out[NODE_PARALLEL],
    FM_TYPE message[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    int layer_num,
    int num_of_nodes
)
{
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<mp_in_t> mp_in[EDGE_PARALLEL][NODE_PARALLEL];
#pragma HLS STREAM variable=mp_in depth=(20 * ceildiv(EMB_DIM, SCATTER_PARALLEL))

    ne_to_mp_adapter(ne_out, mp_in, num_of_nodes);
    for (int pe_id = 0; pe_id < EDGE_PARALLEL; pe_id++)
    {
#pragma HLS UNROLL
        message_passing_pe(
            pe_id,
            layer_num,
            mp_in[pe_id],
            message[pe_id]
        );
    }
}

void ne_to_mp_adapter(
    hls::stream<ne_out_t> ne_out[NODE_PARALLEL],
    hls::stream<mp_in_t> mp_in[EDGE_PARALLEL][NODE_PARALLEL],
    int num_of_nodes
)
{
#pragma HLS INLINE off

#pragma HLS ARRAY_RESHAPE variable=pes_per_node cyclic factor=2 dim=1
#pragma HLS AGGREGATE variable=pes_per_node

    ne_out_t ne_out_struct;
#pragma HLS AGGREGATE variable=ne_out_struct
    mp_in_t mp_in_struct[NODE_PARALLEL];
#pragma HLS AGGREGATE variable=mp_in_struct

    for (int nd_base = 0; nd_base < num_of_nodes; nd_base += NODE_PARALLEL)
    {
#pragma HLS PIPELINE II=ceildiv(EMB_DIM, APPLY_PARALLEL)
#pragma HLS LOOP_TRIPCOUNT min=ceildiv(ANALYSIS_MIN_NODES, NODE_PARALLEL) max=ceildiv(ANALYSIS_MAX_NODES, NODE_PARALLEL) avg=ceildiv(ANALYSIS_AVG_NODES, NODE_PARALLEL)
        for (int ne_dim_base = 0; ne_dim_base < EMB_DIM; ne_dim_base += APPLY_PARALLEL)
        {
            for (int nd_offset = 0; nd_offset < NODE_PARALLEL; nd_offset++)
            {
#pragma HLS UNROLL
                int nd = nd_base + nd_offset;
                if (nd < num_of_nodes)
                {
                    ne_out[nd_offset] >> ne_out_struct;
                    ap_int<EDGE_PARALLEL> node_pes = pes_per_node[nd];
                    for (int ne_dim_offset = 0; ne_dim_offset < APPLY_PARALLEL; ne_dim_offset++)
                    {
#pragma HLS UNROLL
                        int dim = ne_dim_base + ne_dim_offset;
                        if (dim < EMB_DIM)
                        {
                            int mp_dim_offset = dim % SCATTER_PARALLEL;
                            mp_in_struct[nd_offset][mp_dim_offset] = ne_out_struct[ne_dim_offset];
                            if (dim == EMB_DIM - 1 || mp_dim_offset == SCATTER_PARALLEL - 1)
                            {
                                for (int i = 0; i < EDGE_PARALLEL; i++)
                                {
#pragma HLS UNROLL
                                    if (node_pes[i]) mp_in[i][nd_offset] << mp_in_struct[nd_offset];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
