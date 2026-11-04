#include "message_passing.h"
#include "hls_math.h"

typedef struct {
    int nd;
    int degree;
} node_t;

// #region Internal Function Declarations
static void read_degrees(
    int pe_id,
    hls::stream<int>& degrees,
    hls::stream<node_t>& nonzero_degree_nodes,
    int num_of_nodes
);
static void gather(
    int pe_id,
    hls::stream<node_t>& nonzero_degree_nodes,
    FM_VEC h_node[ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_VEC scores_source[MAX_NODE],
    FM_VEC scores_target[ceildiv(MAX_NODE, EDGE_PARALLEL)],
    hls::stream<mp_out_t>& messages,
    hls::stream<FM_VEC>& score_sums
);
static void expand(
    hls::stream<mp_out_t>& messages_per_nz_deg_node,
    hls::stream<FM_VEC>& score_sums_per_nz_deg_node,
    hls::stream<int>& degrees,
    hls::stream<mp_out_t> messages_per_node[NODE_PARALLEL],
    hls::stream<FM_VEC> score_sums_per_node[NODE_PARALLEL],
    int num_of_nodes
);
// #endregion

void message_passing_pe(
    int pe_id,
    FM_VEC h_node[ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_VEC scores_source[MAX_NODE],
    FM_VEC scores_target[ceildiv(MAX_NODE, EDGE_PARALLEL)],
    hls::stream<mp_out_t> messages[NODE_PARALLEL],
    hls::stream<FM_VEC> score_sums[NODE_PARALLEL],
    int num_of_nodes
)
{
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<int> degrees("degrees");
#pragma HLS STREAM variable=degrees depth=20
    hls::stream<node_t> nonzero_degree_nodes("nonzero_degree_nodes");
#pragma HLS STREAM variable=nonzero_degree_nodes depth=20
    hls::stream<mp_out_t> messages_per_nz_deg_node("messages_per_nz_deg_node");
#pragma HLS STREAM variable=messages_per_nz_deg_node depth=(20 * ceildiv(EMB_DIM, GATHER_PARALLEL))
    hls::stream<FM_VEC> score_sums_per_nz_deg_node("score_sums_per_nz_deg_node");
#pragma HLS STREAM variable=score_sums_per_nz_deg_node depth=20

    read_degrees(pe_id, degrees, nonzero_degree_nodes, num_of_nodes);
    gather(pe_id, nonzero_degree_nodes, h_node, scores_source, scores_target, messages_per_nz_deg_node, score_sums_per_nz_deg_node);
    expand(messages_per_nz_deg_node, score_sums_per_nz_deg_node, degrees, messages, score_sums, num_of_nodes);
}

static void read_degrees(
    int pe_id,
    hls::stream<int>& degrees,
    hls::stream<node_t>& nonzero_degree_nodes,
    int num_of_nodes
)
{
#pragma HLS INLINE off

    for (int nd = 0; nd < num_of_nodes; nd++)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODES max=ANALYSIS_MAX_NODES avg=ANALYSIS_AVG_NODES
        int degree = degree_tables[pe_id][nd];
        degrees << degree;
        if (degree != 0)
        {
            nonzero_degree_nodes << node_t{nd, degree};
        }
    }
}

static void gather(
    int pe_id,
    hls::stream<node_t>& nonzero_degree_nodes,
    FM_VEC h_node[ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_VEC scores_source[MAX_NODE],
    FM_VEC scores_target[ceildiv(MAX_NODE, EDGE_PARALLEL)],
    hls::stream<mp_out_t>& messages,
    hls::stream<FM_VEC>& score_sums
)
{
#pragma HLS INLINE off

    mp_out_t mp_outs[ceildiv(EMB_DIM, GATHER_PARALLEL)];
    FM_VEC score_sums_acc;
    int v = 0;
    int e_start = 0;
    int e_end = 0;
    int num_of_edges = num_of_edges_per_pe[pe_id];

    for (int e = 0; e < num_of_edges; e++)
    {
#pragma HLS LOOP_TRIPCOUNT min=0 max=ANALYSIS_MAX_EDGES avg=ceildiv(ANALYSIS_AVG_EDGES, EDGE_PARALLEL)

        int u = neighbor_tables[pe_id][e];

        for (int i = 0, dim_base = 0; i < ceildiv(EMB_DIM, GATHER_PARALLEL); i++, dim_base += GATHER_PARALLEL)
        {
#pragma HLS PIPELINE

            if (e >= e_end)
            {
                node_t node;
                nonzero_degree_nodes >> node;
                v = node.nd;
                e_start = e;
                e_end = e + node.degree;
                score_sums_acc = FM_TYPE(0);
            }

            FM_VEC scores = scores_source[v] + scores_target[u];
            for (int head = 0; head < NUM_HEADS; head++)
            {
#pragma HLS UNROLL
                FM_TYPE score = scores[head];
                if (score < 0) score = score * FM_TYPE(0.2);
                scores[head] = hls::exp(score);
            }
            if (i == 0) score_sums_acc += scores;

            mp_out_t mp_out = (e != e_start) ? mp_outs[i] : FM_VEC(0);
            for (int dim_offset = 0; dim_offset < GATHER_PARALLEL; dim_offset++)
            {
#pragma HLS UNROLL
                int dim = dim_base + dim_offset;
                if (dim < EMB_DIM)
                {
                    mp_out[dim_offset] += scores * h_node[u][dim];
                }
            }
            mp_outs[i] = mp_out;

            if (e + 1 == e_end)
            {
                messages << mp_out;
                if (i == 0) score_sums << score_sums_acc;
            }
        }
    }
}

static void expand(
    hls::stream<mp_out_t>& messages_per_nz_deg_node,
    hls::stream<FM_VEC>& score_sums_per_nz_deg_node,
    hls::stream<int>& degrees,
    hls::stream<mp_out_t> messages_per_node[NODE_PARALLEL],
    hls::stream<FM_VEC> score_sums_per_node[NODE_PARALLEL],
    int num_of_nodes
)
{
#pragma HLS INLINE off

    int degree;
    for (int nd = 0; nd < num_of_nodes; nd++)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODES max=ANALYSIS_MAX_NODES avg=ANALYSIS_AVG_NODES
        for (int i = 0; i < ceildiv(EMB_DIM, GATHER_PARALLEL); i++)
        {
#pragma HLS PIPELINE II=1

            if (i == 0)
            {
#pragma HLS OCCURRENCE cycle=ceildiv(EMB_DIM, GATHER_PARALLEL)
                degrees >> degree;

                FM_VEC score_sum;
                if (degree != 0)
                {
                    score_sums_per_nz_deg_node >> score_sum;
                }
                else
                {
                    score_sum = FM_TYPE(0);
                }
                score_sums_per_node[nd % NODE_PARALLEL] << score_sum;
            }

            mp_out_t message;
            if (degree != 0)
            {
                messages_per_nz_deg_node >> message;
            }
            else
            {
                message = FM_VEC(0);
            }
            messages_per_node[nd % NODE_PARALLEL] << message;
        }
    }
}
