#include "../include/add.hpp"
#include "../include/conv.hpp"
#include "../include/kernel.hpp"
#include "../include/moe.hpp"

void load_one_time_weights(
    wt_patch_embed_t patch_embed_weights_load[FEATURE_DIM][INPUT_CHANNELS][PATCH_HEIGHT][PATCH_WIDTH],
    wt_bias_t patch_embed_bias_load[FEATURE_DIM]
)
{
    #pragma HLS inline off

    {
        hls::vector<wt_bias_t, FEATURE_BLOCK_SIZE>* bias_blocks = reinterpret_cast<hls::vector<wt_bias_t, FEATURE_BLOCK_SIZE>*>(patch_embed_bias_load);
        FOR_BLOCK(dim_out, FEATURE_DIM, FEATURE_BLOCK_SIZE)
        {
            #pragma HLS pipeline

            hls::vector<wt_bias_t, FEATURE_BLOCK_SIZE> bias_block = bias_blocks[dim_out_block];
            FOR_OFFSET(dim_out)
            {
                patch_embed_bias[dim_out] = bias_block[dim_out_offset];
            }
        }
    }

    {
        hls::vector<wt_patch_embed_t, PATCH_WIDTH> (*weights_blocks)[INPUT_CHANNELS][PATCH_HEIGHT] = reinterpret_cast<hls::vector<wt_patch_embed_t, PATCH_WIDTH> (*)[INPUT_CHANNELS][PATCH_HEIGHT]>(patch_embed_weights_load);

        hls::vector<wt_patch_embed_t, PATCH_WIDTH> weights_cache[FEATURE_BLOCK_SIZE];
        #pragma HLS array_partition variable=weights_cache complete dim=1

        FOR_EACH(channel, INPUT_CHANNELS)
        {
            FOR_EACH(y, PATCH_HEIGHT)
            {
                FOR_BLOCK(dim_out, FEATURE_DIM, FEATURE_BLOCK_SIZE)
                {
                    FOR_OFFSET_NOCHK(dim_out)
                    {
                        #pragma HLS pipeline

                        weights_cache[dim_out_offset] = weights_blocks[dim_out][channel][y];

                        if (dim_out_offset == dim_out_step - 1)
                        {
                            #pragma HLS occurrence cycle=dim_out_step

                            FOR_EACH(dim_out_write_offset, dim_out_step)
                            {
                                unsigned int dim_out_write = dim_out_base + dim_out_write_offset;
                                FOR_EACH(x, PATCH_WIDTH)
                                {
                                    patch_embed_weights[dim_out_write][channel][y][x] = weights_cache[dim_out_write_offset][x];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    attn_scale = 0.125;
    norm_eps = 1e-6;
}

extern "C" {
void ViT_compute(
    unsigned int num_images,
    bool reload_one_time_weights,
    image_t images[],
    patch_blocks_t x[],
    patch_blocks_t tmp1,
    patch_blocks_t tmp2,
    patch_blocks_t tmp3,
    patch_blocks_t tmp4,
    fm_block_t tmp_hidden[NUM_PATCHES * ceildiv(max(VIT_HIDDEN_DIM, EXPERT_HIDDEN_DIM), FEATURE_BLOCK_SIZE)],
    qxk_out_t attn,
    softmax_info_t attn_softmax_info,
    wt_patch_embed_t patch_embed_weights[FEATURE_DIM][INPUT_CHANNELS][PATCH_HEIGHT][PATCH_WIDTH],
    wt_bias_t patch_embed_bias[FEATURE_DIM],
    patch_blocks_t pos_embed,
    wt_linear_t attn_weights[NUM_LAYERS][NUM_ATTN_LINEAR][FEATURE_DIM][FEATURE_DIM],
    wt_attn_bias_t attn_bias[NUM_LAYERS][NUM_ATTN_LINEAR][FEATURE_DIM],
    wt_linear_t moe_w_gate[max(NUM_LAYERS / 2, 1U)][NUM_EXPERTS][FEATURE_DIM],
    wt_linear_t moe_weights_l1[max(NUM_LAYERS / 2, 1U)][NUM_EXPERTS][EXPERT_HIDDEN_DIM][FEATURE_DIM],
    wt_bias_t moe_bias_l1[max(NUM_LAYERS / 2, 1U)][NUM_EXPERTS][EXPERT_HIDDEN_DIM],
    wt_linear_t moe_weights_l2[max(NUM_LAYERS / 2, 1U)][NUM_EXPERTS][FEATURE_DIM][EXPERT_HIDDEN_DIM],
    wt_bias_t moe_bias_l2[max(NUM_LAYERS / 2, 1U)][NUM_EXPERTS][FEATURE_DIM],
    wt_linear_t vit_weights_l1[max((NUM_LAYERS + 1) / 2, 1U)][VIT_HIDDEN_DIM][FEATURE_DIM],
    wt_bias_t vit_bias_l1[max((NUM_LAYERS + 1) / 2, 1U)][VIT_HIDDEN_DIM],
    wt_linear_t vit_weights_l2[max((NUM_LAYERS + 1) / 2, 1U)][FEATURE_DIM][VIT_HIDDEN_DIM],
    wt_bias_t vit_bias_l2[max((NUM_LAYERS + 1) / 2, 1U)][FEATURE_DIM],
    wt_norm_t norm_weights[NUM_LAYERS][NUM_LAYER_NORMS][FEATURE_DIM],
    wt_bias_t norm_bias[NUM_LAYERS][NUM_LAYER_NORMS][FEATURE_DIM],
    unsigned int debug_id
)
{
    #pragma HLS interface s_axilite port=return

    #pragma HLS interface m_axi depth=1 port=tmp1 offset=slave bundle=inout1 max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=tmp2 offset=slave bundle=inout2 max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=tmp3 offset=slave bundle=inout3 max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=tmp4 offset=slave bundle=inout4 max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=tmp_hidden offset=slave bundle=inout4 max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=attn offset=slave bundle=inout1 max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=attn_softmax_info offset=slave bundle=inout4 max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=images offset=slave bundle=inout1 max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=x offset=slave bundle=inout2 max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=patch_embed_weights offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=patch_embed_bias offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=pos_embed offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=attn_weights offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=attn_bias offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=moe_w_gate offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=moe_weights_l1 offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=moe_bias_l1 offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=moe_weights_l2 offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=moe_bias_l2 offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=vit_weights_l1 offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=vit_bias_l1 offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=vit_weights_l2 offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=vit_bias_l2 offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=norm_weights offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH
    #pragma HLS interface m_axi depth=1 port=norm_bias offset=slave bundle=weights max_widen_bitwidth=AXI_XFER_BIT_WIDTH

    #pragma HLS allocation function instances=compute_linear limit=1
    #pragma HLS allocation function instances=load_linear_weights limit=1
    #pragma HLS allocation function instances=load_linear_weights<wt_bias_t> limit=1
    #pragma HLS allocation function instances=load_linear_weights<wt_attn_bias_t> limit=1

    if (reload_one_time_weights)
    {
        load_one_time_weights(patch_embed_weights, patch_embed_bias);
    }

    FOR_EACH(image, num_images)
    {
        #pragma HLS loop_tripcount min=1 max=1 avg=1
        #pragma HLS pipeline off

        compute_patch_embed(images[image], x[image], pos_embed);
        if (debug_id == 256) return;

        FOR_EACH(layer, NUM_LAYERS)
        {
            #pragma HLS pipeline off

            load_norms(norm_weights[layer], norm_bias[layer]);
            compute_norm1(x[image], tmp1);
            if (debug_id == layer * 16 + 1) return;
            load_linear_weights(linear_weights_ping, reinterpret_cast<wt_linear_t*>(attn_weights[layer][ATTN_Q]), FEATURE_DIM, FEATURE_DIM);
            load_linear_bias(linear_bias_ping, reinterpret_cast<wt_attn_bias_t*>(attn_bias[layer][ATTN_Q]), FEATURE_DIM);
            compute_linear(reinterpret_cast<fm_block_t*>(tmp2), reinterpret_cast<fm_block_t*>(tmp1), linear_weights_ping, linear_bias_ping, FEATURE_DIM, FEATURE_DIM, 0, false, false, false);
            if (debug_id == layer * 16 + 2) return;
            load_linear_weights(linear_weights_pong, reinterpret_cast<wt_linear_t*>(attn_weights[layer][ATTN_K]), FEATURE_DIM, FEATURE_DIM);
            load_linear_bias(linear_bias_pong, reinterpret_cast<wt_attn_bias_t*>(attn_bias[layer][ATTN_K]), FEATURE_DIM);
            compute_linear(reinterpret_cast<fm_block_t*>(tmp3), reinterpret_cast<fm_block_t*>(tmp1), linear_weights_pong, linear_bias_pong, FEATURE_DIM, FEATURE_DIM, 0, false, false, false);
            if (debug_id == layer * 16 + 3) return;
            compute_q_matmul_k(tmp2, tmp3, attn, attn_softmax_info);
            if (debug_id == layer * 16 + 4) return;
            load_linear_weights(linear_weights_ping, reinterpret_cast<wt_linear_t*>(attn_weights[layer][ATTN_V]), FEATURE_DIM, FEATURE_DIM);
            load_linear_bias(linear_bias_ping, reinterpret_cast<wt_attn_bias_t*>(attn_bias[layer][ATTN_V]), FEATURE_DIM);
            compute_linear(reinterpret_cast<fm_block_t*>(tmp2), reinterpret_cast<fm_block_t*>(tmp1), linear_weights_ping, linear_bias_ping, FEATURE_DIM, FEATURE_DIM, 0, false, false, false);
            if (debug_id == layer * 16 + 5) return;
            compute_attn_matmul_v(tmp2, attn, attn_softmax_info, tmp1);
            if (debug_id == layer * 16 + 6) return;
            load_linear_weights(linear_weights_pong, reinterpret_cast<wt_linear_t*>(attn_weights[layer][ATTN_PROJ]), FEATURE_DIM, FEATURE_DIM);
            load_linear_bias(linear_bias_pong, reinterpret_cast<wt_attn_bias_t*>(attn_bias[layer][ATTN_PROJ]), FEATURE_DIM);
            compute_linear(reinterpret_cast<fm_block_t*>(tmp3), reinterpret_cast<fm_block_t*>(tmp1), linear_weights_pong, linear_bias_pong, FEATURE_DIM, FEATURE_DIM, 0, false, false, false);
            if (debug_id == layer * 16 + 7) return;
            compute_add(x[image], tmp3, x[image]);
            if (debug_id == layer * 16 + 8) return;
            compute_norm2(x[image], tmp1);
            if (debug_id == layer * 16 + 9) return;

            if (layer % 2 == 0)
            {
                load_linear_weights(linear_weights_ping, reinterpret_cast<wt_linear_t*>(vit_weights_l1[layer / 2]), VIT_HIDDEN_DIM, FEATURE_DIM);
                load_linear_bias(linear_bias_ping, reinterpret_cast<wt_bias_t*>(vit_bias_l1[layer / 2]), VIT_HIDDEN_DIM);
                compute_linear(tmp_hidden, reinterpret_cast<fm_block_t*>(tmp1), linear_weights_ping, linear_bias_ping, VIT_HIDDEN_DIM, FEATURE_DIM, 0, true, false, false);
                if (debug_id == layer * 16 + 10) return;
                load_linear_weights(linear_weights_pong, reinterpret_cast<wt_linear_t*>(vit_weights_l2[layer / 2]), FEATURE_DIM, VIT_HIDDEN_DIM);
                load_linear_bias(linear_bias_pong, reinterpret_cast<wt_bias_t*>(vit_bias_l2[layer / 2]), FEATURE_DIM);
                compute_linear(reinterpret_cast<fm_block_t*>(tmp3), tmp_hidden, linear_weights_pong, linear_bias_pong, FEATURE_DIM, VIT_HIDDEN_DIM, 0, false, false, false);
                if (debug_id == layer * 16 + 11) return;
            }
            else
            {
                load_w_gate(moe_w_gate[layer / 2]);
                compute_moe(
                    tmp1,
                    tmp3,
                    tmp_hidden,
                    moe_weights_l1[layer / 2],
                    moe_bias_l1[layer / 2],
                    moe_weights_l2[layer / 2],
                    moe_bias_l2[layer / 2]
                );
                if (debug_id == layer * 16 + 10 || debug_id == layer * 16 + 11) return;
            }
            compute_add(x[image], tmp3, x[image]);
            if (debug_id == layer * 16 + 12) return;
        }
    }
}
}
