#include "../include/attention.hpp"
#include "../include/linear.hpp"
#include <hls_math.h>
#include <hls_stream.h>

static constexpr unsigned int DIM_PER_HEAD = FEATURE_DIM / NUM_HEADS;

fm_t attn_scale;

void read_x(hls::stream<fm_block_t>& x_stream, patch_blocks_t x)
{
    #pragma HLS inline off

    FOR_EACH(patch, NUM_PATCHES)
    {
        FOR_BLOCK(in_dim, FEATURE_DIM, FEATURE_BLOCK_SIZE)
        {
            #pragma HLS pipeline

            x_stream << x[patch][in_dim_block];
        }
    }
}

void read_kv(hls::stream<fm_block_t>& k_stream, patch_blocks_t k)
{
    #pragma HLS inline off

    FOR_BLOCK(q_patch, NUM_PATCHES + ATTN_MATMUL_PARALLEL - 1, ATTN_MATMUL_PARALLEL)
    {
        FOR_EACH(k_patch, NUM_PATCHES)
        {
            FOR_BLOCK(dim, FEATURE_DIM, FEATURE_BLOCK_SIZE)
            {
                #pragma HLS pipeline
                constexpr unsigned int overflow_iters = q_patch_limit % q_patch_step;
                if (overflow_iters > 0 && q_patch_base >= NUM_PATCHES && k_patch >= overflow_iters) return;

                k_stream << k[k_patch][dim_block];
            }
        }
    }
}

void compute_q_matmul_k(
    hls::stream<attn_parallel_t>& attn_stream,
    hls::stream<fm_block_t>& q_stream,
    hls::stream<fm_block_t>& k_stream
)
{
    #pragma HLS inline off

    fm_blocks_t q_blocks[ATTN_MATMUL_PARALLEL];
    #pragma HLS array_partition variable=q_blocks complete dim=1

    fm_t attn_blocks[ATTN_MATMUL_PARALLEL][NUM_HEADS];
    #pragma HLS array_partition variable=attn_blocks complete dim=1
    #pragma HLS array_partition variable=attn_blocks complete dim=2

    static_assert(ATTN_MATMUL_PARALLEL < NUM_PATCHES, "ATTN_MATMUL_PARALLEL must be less than NUM_PATCHES");

    FOR_BLOCK(q_patch, NUM_PATCHES + ATTN_MATMUL_PARALLEL - 1, ATTN_MATMUL_PARALLEL)
    {
        FOR_EACH(k_patch, NUM_PATCHES)
        {
            FOR_BLOCK(dim, FEATURE_DIM, FEATURE_BLOCK_SIZE)
            {
                #pragma HLS pipeline
                constexpr unsigned int overflow_iters = q_patch_limit % q_patch_step;
                if (overflow_iters > 0 && q_patch_base >= NUM_PATCHES && k_patch >= overflow_iters) return;

                if (k_patch < ATTN_MATMUL_PARALLEL && q_patch_base + k_patch < NUM_PATCHES)
                {
                    q_stream >> q_blocks[k_patch][dim_block];
                }

                fm_block_t k_block;
                k_stream >> k_block;

                if (dim_block == 0)
                {
                    #pragma HLS occurrence cycle=dim_iters

                    FOR_OFFSET_UNSAFE(q_patch)
                    {
                        FOR_EACH(head, NUM_HEADS)
                        {
                            attn_blocks[q_patch_offset][head] = 0;
                        }
                    }
                }

                static_assert(DIM_PER_HEAD % FEATURE_BLOCK_SIZE == 0, "DIM_PER_HEAD must be a multiple of FEATURE_BLOCK_SIZE");
                unsigned int head = dim_block / (DIM_PER_HEAD / FEATURE_BLOCK_SIZE);
                FOR_OFFSET_UNSAFE(q_patch)
                {
                    fm_block_t q_block = q_blocks[q_patch_offset][dim_block];

                    fm_t attn_addend = 0;
                    FOR_OFFSET(dim)
                    {
                        attn_addend += q_block[dim_offset] * k_block[dim_offset];
                    }
                    attn_addend *= attn_scale;
                    attn_blocks[q_patch_offset][head] += attn_addend;
                }

                if (dim_block == dim_iters - 1)
                {
                    #pragma HLS occurrence cycle=dim_iters

                    attn_parallel_t finalized;
                    FOR_OFFSET_UNSAFE(q_patch)
                    {
                        FOR_EACH(head, NUM_HEADS)
                        {
                            finalized[q_patch_offset][head] = attn_blocks[q_patch_offset][head];
                        }
                    }
                    attn_stream << finalized;
                }
            }
        }
    }
}

void finalize_attn(
    hls::stream<attn_parallel_t>& qxk_stream,
    hls::stream<heads_t>& attn_stream,
    hls::stream<softmax_info_row_t>& attn_softmax_info_stream
)
{
    #pragma HLS inline off

    fm_t softmax_sums[ATTN_MATMUL_PARALLEL][NUM_HEADS];
    #pragma HLS array_partition variable=softmax_sums complete dim=1
    #pragma HLS array_partition variable=softmax_sums complete dim=2
    fm_t softmax_biases[ATTN_MATMUL_PARALLEL][NUM_HEADS];
    #pragma HLS array_partition variable=softmax_biases complete dim=1
    #pragma HLS array_partition variable=softmax_biases complete dim=2

    heads_t attn_blocks[ATTN_MATMUL_PARALLEL];
    #pragma HLS array_partition variable=attn_blocks complete dim=1

    FOR_BLOCK(q_patch_unadjusted, NUM_PATCHES + ATTN_MATMUL_PARALLEL - 1, ATTN_MATMUL_PARALLEL)
    {
        FOR_EACH(k_patch, NUM_PATCHES)
        {
            FOR_OFFSET_UNSAFE(q_patch_unadjusted)
            {
                #pragma HLS pipeline
                #pragma HLS dependence variable=softmax_sums inter true distance=q_patch_unadjusted_step
                #pragma HLS dependence variable=softmax_biases inter true distance=q_patch_unadjusted_step
                constexpr unsigned int overflow_iters = q_patch_unadjusted_limit % q_patch_unadjusted_step;
                if (overflow_iters > 0 && q_patch_unadjusted_base >= NUM_PATCHES && k_patch >= overflow_iters) return;

                if (q_patch_unadjusted_offset == 0)
                {
                    #pragma HLS occurrence cycle=q_patch_unadjusted_step
                    attn_parallel_t attn_blocks_packed;
                    qxk_stream >> attn_blocks_packed;
                    FOR_EACH(unpack_idx, ATTN_MATMUL_PARALLEL)
                    {
                        attn_blocks[unpack_idx] = attn_blocks_packed[unpack_idx];
                    }
                }

                softmax_info_row_t softmax_info_row;
                FOR_EACH(head, NUM_HEADS)
                {
                    fm_t attn_head = attn_blocks[q_patch_unadjusted_offset][head];
                    fm_t curr_softmax_sum = 0.0;
                    fm_t curr_softmax_bias = ap_fixed_min<fm_t>();
                    if (k_patch != q_patch_unadjusted_offset)
                    {
                        curr_softmax_sum = softmax_sums[q_patch_unadjusted_offset][head];
                        curr_softmax_bias = softmax_biases[q_patch_unadjusted_offset][head];
                    }

                    auto exp_arg_noclip = curr_softmax_bias - attn_head;
                    bool is_new_bias = hls::signbit(exp_arg_noclip);
                    fm_t exp_arg = (is_new_bias) ? fm_t(exp_arg_noclip) : fm_t(-exp_arg_noclip);
                    fm_t exp = hls::exp(exp_arg);
                    if (is_new_bias)
                    {
                        curr_softmax_sum *= exp;
                        curr_softmax_sum += 1;
                        curr_softmax_bias = attn_head;
                    }
                    else
                    {
                        curr_softmax_sum += exp;
                    }

                    fm_t curr_softmax_sum_recip = hls::recip(curr_softmax_sum);
                    softmax_sums[q_patch_unadjusted_offset][head] = curr_softmax_sum;
                    softmax_biases[q_patch_unadjusted_offset][head] = curr_softmax_bias;
                    softmax_info_row[head * 2] = curr_softmax_sum_recip;
                    softmax_info_row[head * 2 + 1] = curr_softmax_bias;
                }

                unsigned int q_patch = (k_patch >= q_patch_unadjusted_offset) ? q_patch_unadjusted : q_patch_unadjusted - ATTN_MATMUL_PARALLEL;
                bool q_patch_valid = ((k_patch >= q_patch_unadjusted_offset) || (q_patch_unadjusted_block > 0)) && (q_patch < NUM_PATCHES);
                if (q_patch_valid && k_patch == ((q_patch_unadjusted_offset == 0) ? (NUM_PATCHES - 1) : (q_patch_unadjusted_offset - 1)))
                {
                    #pragma HLS occurrence cycle=(NUM_PATCHES + 1)
                    attn_softmax_info_stream << softmax_info_row;
                }
                attn_stream << attn_blocks[q_patch_unadjusted_offset];
            }
        }
    }
}

void write_attn(
    hls::stream<heads_t>& attn_stream,
    qxk_out_t attn
)
{
    #pragma HLS inline off

    constexpr unsigned int q_patch_limit = NUM_PATCHES + ATTN_MATMUL_PARALLEL - 1;
    constexpr unsigned int q_patch_step = ATTN_MATMUL_PARALLEL;
    constexpr unsigned int q_patch_block_limit = q_patch_limit / q_patch_step;
    constexpr unsigned int overflow_iters = q_patch_limit % q_patch_step;

    FOR_EACH(q_patch_block, q_patch_block_limit)
    {
        FOR_EACH(k_patch, NUM_PATCHES)
        {
            FOR_EACH(q_patch_offset, q_patch_step)
            {
                #pragma HLS pipeline
                attn_stream >> attn[q_patch_block][k_patch][q_patch_offset];
            }
        }
    }
    FOR_EACH(k_patch, overflow_iters)
    {
        FOR_EACH(q_patch_offset, q_patch_step)
        {
            #pragma HLS pipeline
            attn_stream >> attn[q_patch_block_limit][k_patch][q_patch_offset];
        }
    }
}

void write_attn_softmax_info(
    hls::stream<softmax_info_row_t>& attn_softmax_info_stream,
    softmax_info_t attn_softmax_info
)
{
    #pragma HLS inline off

    FOR_EACH(q_patch, NUM_PATCHES)
    {
        #pragma HLS pipeline
        attn_softmax_info_stream >> attn_softmax_info[q_patch];
    }
}

void compute_q_matmul_k(
    patch_blocks_t q,
    patch_blocks_t k,
    qxk_out_t attn,
    softmax_info_t attn_softmax_info
)
{
    #pragma HLS inline off
    #pragma HLS dataflow

    hls::stream<fm_block_t> q_stream("q_stream");
    hls::stream<fm_block_t> k_stream("k_stream");
    hls::stream<attn_parallel_t> qxk_stream("qxk_stream");
    hls::stream<heads_t> attn_stream("attn_stream");
    hls::stream<softmax_info_row_t> attn_softmax_info_stream("attn_softmax_info_stream");

    read_x(q_stream, q);
    read_kv(k_stream, k);
    compute_q_matmul_k(qxk_stream, q_stream, k_stream);
    finalize_attn(qxk_stream, attn_stream, attn_softmax_info_stream);
    write_attn(attn_stream, attn);
    write_attn_softmax_info(attn_softmax_info_stream, attn_softmax_info);
}

void read_attn(
    hls::stream<heads_t>& attn_stream,
    qxk_out_t attn
)
{
    #pragma HLS inline off

    constexpr unsigned int q_patch_limit = NUM_PATCHES + ATTN_MATMUL_PARALLEL - 1;
    constexpr unsigned int q_patch_step = ATTN_MATMUL_PARALLEL;
    constexpr unsigned int q_patch_block_limit = q_patch_limit / q_patch_step;
    constexpr unsigned int overflow_iters = q_patch_limit % q_patch_step;

    FOR_EACH(q_patch_block, q_patch_block_limit)
    {
        FOR_EACH(k_patch, NUM_PATCHES)
        {
            FOR_EACH(q_patch_offset, q_patch_step)
            {
                #pragma HLS pipeline
                attn_stream << attn[q_patch_block][k_patch][q_patch_offset];
            }
        }
    }
    FOR_EACH(k_patch, overflow_iters)
    {
        FOR_EACH(q_patch_offset, q_patch_step)
        {
            #pragma HLS pipeline
            attn_stream << attn[q_patch_block_limit][k_patch][q_patch_offset];
        }
    }
}

void prepare_attn(
    hls::stream<heads_t>& attn_stream,
    hls::stream<attn_parallel_t>& qxk_out_stream
)
{
    #pragma HLS inline off

    heads_t attn_blocks[ATTN_MATMUL_PARALLEL];
    #pragma HLS array_partition variable=attn_blocks complete dim=1

    FOR_BLOCK(q_patch, NUM_PATCHES + ATTN_MATMUL_PARALLEL - 1, ATTN_MATMUL_PARALLEL)
    {
        FOR_EACH(k_patch, NUM_PATCHES)
        {
            FOR_OFFSET_UNSAFE(q_patch)
            {
                #pragma HLS pipeline
                constexpr unsigned int overflow_iters = q_patch_limit % q_patch_step;
                if (overflow_iters > 0 && q_patch_base >= NUM_PATCHES && k_patch >= overflow_iters) return;

                attn_stream >> attn_blocks[q_patch_offset];
                if (q_patch_offset == q_patch_step - 1)
                {
                    #pragma HLS occurrence cycle=q_patch_step
                    attn_parallel_t attn_packed;
                    FOR_EACH(pack_idx, ATTN_MATMUL_PARALLEL)
                    {
                        attn_packed[pack_idx] = attn_blocks[pack_idx];
                    }
                    qxk_out_stream << attn_packed;
                }
            }
        }
    }
}

void read_attn_softmax_info(
    hls::stream<softmax_info_row_t>& attn_softmax_info_stream,
    softmax_info_t attn_softmax_info
)
{
    #pragma HLS inline off

    FOR_EACH(q_patch, NUM_PATCHES)
    {
        #pragma HLS pipeline
        attn_softmax_info_stream << attn_softmax_info[q_patch];
    }
}

void write_attn_matmul_v(
    patch_blocks_t attn_matmul_v,
    hls::stream<fm_block_t>& attn_matmul_v_stream
)
{
    #pragma HLS inline off

    FOR_EACH(patch, NUM_PATCHES)
    {
        FOR_BLOCK(dim, FEATURE_DIM, FEATURE_BLOCK_SIZE)
        {
            #pragma HLS pipeline

            attn_matmul_v_stream >> attn_matmul_v[patch][dim_block];
        }
    }
}

void compute_attn_matmul_v(
    hls::stream<fm_block_t>& attn_matmul_v_stream,
    hls::stream<attn_parallel_t>& qxk_out_stream,
    hls::stream<softmax_info_row_t>& attn_softmax_info_stream,
    hls::stream<fm_block_t>& v_stream
)
{
    #pragma HLS inline off

    fm_blocks_t acc_blocks[ATTN_MATMUL_PARALLEL];
    #pragma HLS array_partition variable=acc_blocks complete dim=1
    fm_t attn[ATTN_MATMUL_PARALLEL][NUM_HEADS];
    #pragma HLS array_partition variable=attn complete dim=1
    #pragma HLS array_partition variable=attn complete dim=2
    softmax_info_row_t attn_softmax_info[ATTN_MATMUL_PARALLEL];
    #pragma HLS array_partition variable=attn_softmax_info complete dim=1

    FOR_BLOCK(attn_patch, NUM_PATCHES + ATTN_MATMUL_PARALLEL - 1, ATTN_MATMUL_PARALLEL)
    {
        FOR_EACH(v_patch, NUM_PATCHES)
        {
            FOR_BLOCK(dim, FEATURE_DIM, FEATURE_BLOCK_SIZE)
            {
                #pragma HLS pipeline
                constexpr unsigned int overflow_iters = attn_patch_limit % attn_patch_step;
                if (overflow_iters > 0 && attn_patch_base >= NUM_PATCHES && v_patch >= overflow_iters) return;

                if (dim_block == 0)
                {
                    #pragma HLS occurrence cycle=dim_iters

                    if (v_patch < ATTN_MATMUL_PARALLEL && attn_patch_base + v_patch < NUM_PATCHES)
                    {
                        attn_softmax_info_stream >> attn_softmax_info[v_patch];
                    }

                    attn_parallel_t attn_packed;
                    qxk_out_stream >> attn_packed;
                    FOR_OFFSET_UNSAFE(attn_patch)
                    {
                        FOR_EACH(head, NUM_HEADS)
                        {
                            fm_t softmax_sum_recip = attn_softmax_info[attn_patch_offset][head * 2];
                            fm_t softmax_bias = attn_softmax_info[attn_patch_offset][head * 2 + 1];
                            attn[attn_patch_offset][head] = (
                                hls::exp(fm_t(attn_packed[attn_patch_offset][head] - softmax_bias))
                                * softmax_sum_recip
                            );
                        }
                    }
                }

                fm_block_t v_block;
                v_stream >> v_block;

                static_assert(DIM_PER_HEAD % FEATURE_BLOCK_SIZE == 0, "DIM_PER_HEAD must be a multiple of FEATURE_BLOCK_SIZE");
                unsigned int head = dim_block / (DIM_PER_HEAD / FEATURE_BLOCK_SIZE);
                FOR_OFFSET_UNSAFE(attn_patch)
                {
                    fm_block_t acc_block;
                    FOR_OFFSET(dim)
                    {
                        acc_block[dim_offset] = v_block[dim_offset] * attn[attn_patch_offset][head];
                    }

                    if (v_patch != attn_patch_offset)
                    {
                        acc_block += acc_blocks[attn_patch_offset][dim_block];
                    }
                    acc_blocks[attn_patch_offset][dim_block] = acc_block;
                }

                if (v_patch == NUM_PATCHES - 1)
                {
                    attn_matmul_v_stream << acc_blocks[0][dim_block];
                }
                else if (v_patch < ATTN_MATMUL_PARALLEL - 1 && attn_patch_block > 0)
                {
                    attn_matmul_v_stream << acc_blocks[v_patch + 1][dim_block];
                }
            }
        }
    }
}

void compute_attn_matmul_v(
    patch_blocks_t v,
    qxk_out_t attn,
    softmax_info_t attn_softmax_info,
    patch_blocks_t attn_matmul_v
)
{
    #pragma HLS inline off
    #pragma HLS dataflow

    hls::stream<fm_block_t> attn_matmul_v_stream("attn_matmul_v_stream");
    hls::stream<heads_t> attn_stream("attn_stream");
    hls::stream<attn_parallel_t> qxk_out_stream("qxk_out_stream");
    hls::stream<softmax_info_row_t> attn_softmax_info_stream("attn_softmax_info_stream");
    hls::stream<fm_block_t> v_stream("v_stream");

    read_kv(v_stream, v);
    read_attn(attn_stream, attn);
    read_attn_softmax_info(attn_softmax_info_stream, attn_softmax_info);
    prepare_attn(attn_stream, qxk_out_stream);
    compute_attn_matmul_v(attn_matmul_v_stream, qxk_out_stream, attn_softmax_info_stream, v_stream);
    write_attn_matmul_v(attn_matmul_v, attn_matmul_v_stream);
}
