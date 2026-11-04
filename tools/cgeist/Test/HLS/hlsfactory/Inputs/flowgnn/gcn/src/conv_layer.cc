#include "conv_layer.h"
#include "node_embedding.h"
#include "message_passing.h"
#include "load_inputs.h"
#include "finalize.h"
#include "hls_stream.h"

// #region Internal Function Declarations
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
    int num_of_nodes
)
{
#pragma HLS INLINE off
#pragma HLS DATAFLOW

#pragma HLS ARRAY_PARTITION variable=message cyclic factor=SCATTER_PARALLEL dim=3
#pragma HLS ARRAY_PARTITION variable=next_message cyclic factor=SCATTER_PARALLEL dim=3

    hls::stream<ne_out_t> embeddings[NODE_PARALLEL];
#pragma HLS STREAM variable=embeddings depth=200

    node_embedding_multi_pe(embeddings, message, layer_num, num_of_nodes);
    message_passing_all_pes(embeddings, next_message, layer_num, num_of_nodes);
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

#pragma HLS ARRAY_PARTITION variable=message complete dim=1

    hls::stream<mp_in_t> node_embeddings[EDGE_PARALLEL][NODE_PARALLEL];
#pragma HLS STREAM variable=node_embeddings depth=40

    ne_to_mp_adapter(ne_out, node_embeddings, num_of_nodes);
    for (int pe_id = 0; pe_id < EDGE_PARALLEL; pe_id++)
    {
#pragma HLS UNROLL
        message_passing_pe(
            pe_id,
            node_embeddings[pe_id],
            message[pe_id],
            layer_num,
            num_of_nodes
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
                                    mp_in[i][nd_offset] << mp_in_struct[nd_offset];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
