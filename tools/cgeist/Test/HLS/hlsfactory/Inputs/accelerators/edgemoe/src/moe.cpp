#include "../include/add.hpp"
#include "../include/moe.hpp"
#include "../include/linear.hpp"
#include <hls_math.h>
#include <hls_stream.h>

typedef hls::vector<fm_t, NUM_EXPERTS> experts_block_t;
typedef hls::vector<unsigned int, NUM_SELECTED_EXPERTS> topk_indices_t;
typedef hls::vector<fm_t, NUM_SELECTED_EXPERTS> topk_scores_t;

wt_linear_t w_gate[NUM_EXPERTS][FEATURE_DIM];

unsigned int expert_queues[NUM_EXPERTS][NUM_PATCHES];
fm_t expert_scores[NUM_EXPERTS][NUM_PATCHES];
unsigned int expert_queue_lens[NUM_EXPERTS];
unsigned int expert_metaqueue[NUM_EXPERTS];
unsigned int expert_metaqueue_len;

void load_w_gate(wt_linear_t w_gate_load[NUM_EXPERTS][FEATURE_DIM])
{
    #pragma HLS inline off

    hls::vector<wt_linear_t, FEATURE_BLOCK_SIZE> (*w_gate_blocks)[FEATURE_DIM / FEATURE_BLOCK_SIZE] = reinterpret_cast<hls::vector<wt_linear_t, FEATURE_BLOCK_SIZE> (*)[FEATURE_DIM / FEATURE_BLOCK_SIZE]>(w_gate_load);

    hls::vector<wt_linear_t, FEATURE_BLOCK_SIZE> w_gate_cache[NUM_EXPERTS];
    #pragma HLS array_partition variable=w_gate_cache complete dim=1

    FOR_BLOCK(expert, NUM_EXPERTS, NUM_EXPERTS)
    {
        FOR_BLOCK(dim, FEATURE_DIM, FEATURE_BLOCK_SIZE)
        {
            FOR_OFFSET_NOCHK(expert)
            {
                #pragma HLS pipeline

                w_gate_cache[expert_offset] = w_gate_blocks[expert][dim_block];

                if (expert_offset == expert_step - 1)
                {
                    #pragma HLS occurrence cycle=expert_step

                    FOR_EACH(expert_write_offset, expert_step)
                    {
                        unsigned int expert_write = expert_base + expert_write_offset;
                        FOR_OFFSET(dim)
                        {
                            w_gate[expert_write][dim] = w_gate_cache[expert_write_offset][dim_offset];
                        }
                    }
                }
            }
        }
    }
}

void read_gate_inp(hls::stream<fm_block_t>& gate_inp_stream, patch_blocks_t gate_inp)
{
    #pragma HLS inline off

    FOR_EACH(patch, NUM_PATCHES)
    {
        FOR_BLOCK(out_dim, NUM_EXPERTS, NUM_EXPERTS)
        {
            FOR_BLOCK(in_dim, FEATURE_DIM, FEATURE_BLOCK_SIZE)
            {
                #pragma HLS pipeline

                gate_inp_stream << gate_inp[patch][in_dim_block];
            }
        }
    }
}

void write_gate_results(
    hls::stream<topk_indices_t>& topk_indices_stream,
    hls::stream<topk_scores_t>& topk_scores_stream,
    unsigned int expert_queues[NUM_EXPERTS][NUM_PATCHES],
    fm_t expert_scores[NUM_EXPERTS][NUM_PATCHES],
    unsigned int expert_queue_lens[NUM_EXPERTS]
)
{
    #pragma HLS inline off

    #pragma HLS array_partition variable=expert_queues complete dim=1
    #pragma HLS array_partition variable=expert_scores complete dim=1
    #pragma HLS array_partition variable=expert_queue_lens complete dim=1

    FOR_EACH(expert, NUM_EXPERTS)
    {
        #pragma HLS unroll

        expert_queue_lens[expert] = 0;
    }

    FOR_EACH(patch, NUM_PATCHES)
    {
        #pragma HLS pipeline

        topk_indices_t topk_indices;
        topk_scores_t topk_scores;
        topk_indices_stream >> topk_indices;
        topk_scores_stream >> topk_scores;

        FOR_EACH(expert_index, NUM_SELECTED_EXPERTS)
        {
            unsigned int expert = topk_indices[expert_index];
            unsigned int queue_index = expert_queue_lens[expert];
            expert_queue_lens[expert] = queue_index + 1;

            expert_queues[expert][queue_index] = patch;
            expert_scores[expert][queue_index] = topk_scores[expert_index];
        }
    }

    expert_metaqueue_len = 0;
    FOR_EACH(expert, NUM_EXPERTS)
    {
        #pragma HLS unroll

        if (expert_queue_lens[expert] > 0)
        {
            expert_metaqueue[expert_metaqueue_len] = expert;
            expert_metaqueue_len++;
        }
    }
}

void top_k(
    topk_indices_t& indices,
    topk_scores_t& scores,
    experts_block_t& update
)
{
    #pragma HLS inline off

    FOR_EACH(i, NUM_EXPERTS)
    {
        #pragma HLS pipeline rewind

        topk_indices_t prev_indices = indices;
        topk_scores_t prev_scores = scores;
        topk_indices_t new_indices;
        topk_scores_t new_scores;
        fm_t score = update[i];
        bool shift = false;

        FOR_EACH(j, NUM_SELECTED_EXPERTS)
        {
            #pragma HLS unroll

            if (shift)
            {
                new_scores[j] = prev_scores[j - 1];
                new_indices[j] = prev_indices[j - 1];
            }
            else if (score > prev_scores[j])
            {
                new_scores[j] = score;
                new_indices[j] = i;
                shift = true;
            }
        }

        indices = new_indices;
        scores = new_scores;
    }
}

experts_block_t compute_gating_for_patch(hls::stream<fm_block_t>& gate_inp_stream)
{
    #pragma HLS inline off

    hls::vector<fm_t, NUM_EXPERTS> scores = fm_t(0);

    FOR_BLOCK(in_dim, FEATURE_DIM, FEATURE_BLOCK_SIZE)
    {
        #pragma HLS pipeline rewind

        experts_block_t addend = fm_t(0);
        fm_block_t gate_inp_block;
        gate_inp_stream >> gate_inp_block;
        FOR_EACH(out_dim, NUM_EXPERTS)
        {
            #pragma HLS unroll
            FOR_OFFSET_NOCHK(in_dim)
            {
                #pragma HLS unroll
                addend[out_dim] += gate_inp_block[in_dim_offset] * w_gate[out_dim][in_dim];
            }
        }
        scores += addend;
    }

    return scores;
}

void update_softmax_info(
    fm_t& scores_softmax_sum,
    fm_t& scores_softmax_bias,
    experts_block_t scores
)
{
    #pragma HLS inline off

    FOR_EACH(out_dim, NUM_EXPERTS)
    {
        #pragma HLS pipeline rewind II=3

        fm_t score = scores[out_dim];
        auto exp_arg_noclip = scores_softmax_bias - score;
        bool is_new_bias = hls::signbit(exp_arg_noclip);
        fm_t exp_arg = (is_new_bias) ? fm_t(exp_arg_noclip) : fm_t(-exp_arg_noclip);
        fm_t exp = hls::exp(exp_arg);
        if (is_new_bias)
        {
            scores_softmax_sum *= exp;
            scores_softmax_sum += 1;
            scores_softmax_bias = score;
        }
        else
        {
            scores_softmax_sum += exp;
        }
    }
}

topk_scores_t finalize_topk_scores_softmax(
    topk_scores_t top_k_scores,
    fm_t scores_softmax_sum,
    fm_t scores_softmax_bias
)
{
    #pragma HLS inline off

    topk_scores_t top_k_scores_softmax;
    FOR_EACH(expert, NUM_SELECTED_EXPERTS)
    {
        #pragma HLS unroll
        top_k_scores_softmax[expert] = hls::exp(fm_t(top_k_scores[expert] - scores_softmax_bias)) / scores_softmax_sum;
    }
    return top_k_scores_softmax;
}

void compute_gating(
    hls::stream<fm_block_t>& gate_inp_stream,
    hls::stream<topk_indices_t>& topk_indices_stream,
    hls::stream<topk_scores_t>& topk_scores_stream
)
{
    #pragma HLS inline off

    #pragma HLS array_reshape variable=w_gate cyclic factor=NUM_EXPERTS dim=1
    #pragma HLS array_reshape variable=w_gate cyclic factor=FEATURE_BLOCK_SIZE dim=2

    FOR_EACH(patch, NUM_PATCHES)
    {
        topk_indices_t top_k_indices = 0;
        topk_scores_t top_k_scores = ap_fixed_min<fm_t>();
        fm_t scores_softmax_sum = 0;
        fm_t scores_softmax_bias = ap_fixed_min<fm_t>();

        hls::vector<fm_t, NUM_EXPERTS> scores = compute_gating_for_patch(gate_inp_stream);
        update_softmax_info(scores_softmax_sum, scores_softmax_bias, scores);
        top_k(top_k_indices, top_k_scores, scores);

        topk_scores_t top_k_scores_softmax = finalize_topk_scores_softmax(top_k_scores, scores_softmax_sum, scores_softmax_bias);
        topk_indices_stream << top_k_indices;
        topk_scores_stream << top_k_scores_softmax;
    }
}

void compute_gating(
    patch_blocks_t gate_inp,
    unsigned int expert_queues[NUM_EXPERTS][NUM_PATCHES],
    fm_t expert_scores[NUM_EXPERTS][NUM_PATCHES],
    unsigned int expert_queue_lens[NUM_EXPERTS]
)
{
    #pragma HLS inline off
    #pragma HLS dataflow

    hls::stream<fm_block_t> gate_inp_stream;
    hls::stream<topk_indices_t> topk_indices_stream;
    hls::stream<topk_scores_t> topk_scores_stream;

    read_gate_inp(gate_inp_stream, gate_inp);
    compute_gating(gate_inp_stream, topk_indices_stream, topk_scores_stream);
    write_gate_results(topk_indices_stream, topk_scores_stream, expert_queues, expert_scores, expert_queue_lens);
}

void zero_output(patch_blocks_t out)
{
    #pragma HLS inline off

    FOR_EACH(patch, NUM_PATCHES)
    {
        FOR_BLOCK(dim, FEATURE_DIM, FEATURE_BLOCK_SIZE)
        {
            out[patch][dim_block] = fm_t(0);
        }
    }
}

void compute_moe(
    patch_blocks_t gate_inp,
    patch_blocks_t out,
    fm_block_t tmp_hidden[],
    wt_linear_t all_weights_l1[NUM_EXPERTS][EXPERT_HIDDEN_DIM][FEATURE_DIM],
    wt_bias_t all_bias_l1[NUM_EXPERTS][EXPERT_HIDDEN_DIM],
    wt_linear_t all_weights_l2[NUM_EXPERTS][FEATURE_DIM][EXPERT_HIDDEN_DIM],
    wt_bias_t all_bias_l2[NUM_EXPERTS][FEATURE_DIM]
)
{
    #pragma HLS inline

    compute_gating(gate_inp, expert_queues, expert_scores, expert_queue_lens);
    zero_output(out);

    unsigned int expert_load = expert_metaqueue[0];
    if (NUM_SELECTED_EXPERTS > 0 || expert_metaqueue_len > 0)
    {
        load_linear_weights(
            linear_weights_ping,
            reinterpret_cast<wt_linear_t*>(all_weights_l1[expert_load]),
            EXPERT_HIDDEN_DIM,
            FEATURE_DIM
        );
        load_linear_bias(
            linear_bias_ping,
            reinterpret_cast<wt_bias_t*>(all_bias_l1[expert_load]),
            EXPERT_HIDDEN_DIM
        );
    }

    unsigned int expert_compute = expert_load;
    _LABEL_FOR_EACH(__LINE__, expert_load_idx): for (
        unsigned int expert_load_idx = 1;
        expert_load_idx < expert_metaqueue_len;
        expert_load_idx++
    )
    {
        #pragma HLS pipeline off
        #pragma HLS loop_tripcount min=0 avg=(NUM_EXPERTS - 1) max=(NUM_EXPERTS - 1)

        expert_compute = expert_load;
        compute_linear(
            tmp_hidden,
            reinterpret_cast<fm_block_t*>(gate_inp),
            linear_weights_ping,
            linear_bias_ping,
            EXPERT_HIDDEN_DIM,
            FEATURE_DIM,
            expert_compute,
            true,
            true,
            false
        );
        load_linear_weights(
            linear_weights_pong,
            reinterpret_cast<wt_linear_t*>(all_weights_l2[expert_compute]),
            FEATURE_DIM,
            EXPERT_HIDDEN_DIM
        );
        load_linear_bias(
            linear_bias_pong,
            reinterpret_cast<wt_bias_t*>(all_bias_l2[expert_compute]),
            FEATURE_DIM
        );

        expert_load = expert_metaqueue[expert_load_idx];
        compute_linear(
            reinterpret_cast<fm_block_t*>(out),
            tmp_hidden,
            linear_weights_pong,
            linear_bias_pong,
            FEATURE_DIM,
            EXPERT_HIDDEN_DIM,
            expert_compute,
            false,
            true,
            true
        );
        load_linear_weights(
            linear_weights_ping,
            reinterpret_cast<wt_linear_t*>(all_weights_l1[expert_load]),
            EXPERT_HIDDEN_DIM,
            FEATURE_DIM
        );
        load_linear_bias(
            linear_bias_ping,
            reinterpret_cast<wt_bias_t*>(all_bias_l1[expert_load]),
            EXPERT_HIDDEN_DIM
        );
    }

    if (NUM_SELECTED_EXPERTS > 0 || expert_metaqueue_len > 0)
    {
        expert_compute = expert_load;
        compute_linear(
            tmp_hidden,
            reinterpret_cast<fm_block_t*>(gate_inp),
            linear_weights_ping,
            linear_bias_ping,
            EXPERT_HIDDEN_DIM,
            FEATURE_DIM,
            expert_compute,
            true,
            true,
            false
        );
        load_linear_weights(
            linear_weights_pong,
            reinterpret_cast<wt_linear_t*>(all_weights_l2[expert_compute]),
            FEATURE_DIM,
            EXPERT_HIDDEN_DIM
        );
        load_linear_bias(
            linear_bias_pong,
            reinterpret_cast<wt_bias_t*>(all_bias_l2[expert_compute]),
            FEATURE_DIM
        );

        compute_linear(
            reinterpret_cast<fm_block_t*>(out),
            tmp_hidden,
            linear_weights_pong,
            linear_bias_pong,
            FEATURE_DIM,
            EXPERT_HIDDEN_DIM,
            expert_compute,
            false,
            true,
            true
        );
    }
}
