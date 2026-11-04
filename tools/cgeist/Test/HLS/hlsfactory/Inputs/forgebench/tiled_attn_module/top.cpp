
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>
#include <hls_math.h>
using namespace std;

typedef ap_fixed<16, 5> data_t;

data_t BRAM_attn_input[8][256];
data_t BRAM_weights_q[64][256];
data_t BRAM_weights_k[64][256];
data_t BRAM_weights_v[64][256];
data_t BRAM_1[8][64];

void load_8_256_ap_fixed_16_5_(data_t input[8][256], data_t output[8][256])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 8; idx0++) {
        for (int idx1 = 0; idx1 < 256; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_64_256_ap_fixed_16_5_(data_t input[64][256], data_t output[64][256])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 64; idx0++) {
        for (int idx1 = 0; idx1 < 256; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_256_256_ap_fixed_16_5_(data_t input[256][256], data_t output[256][256])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 256; idx0++) {
        for (int idx1 = 0; idx1 < 256; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

/*
 * Auto-generated Grouped Multi-head Attention (with optional inline RoPE)
 *
 * Input     : [8][256]
 * W_q/k/v   : [64][256], DIM_OUT = NUM_HEADS * HEAD_DIM
 * Output    : [8][64]
 *
 * Data type : ap_fixed<16, 5>
 * Num Heads : 4
 * Head Dim  : 16
 */

 void transpose_64(
    ap_fixed<16, 5> input[64][64]
)
{
    #pragma HLS inline off
for (int i = 0; i < 64; i++) {
for (int j = 0; j < 64; j++) {
    input[i][j] = input[j][i];
}
}
}

void transpoe_256(
    ap_fixed<16, 5> input[256][256]
)
{
    #pragma HLS inline off
for (int i = 0; i < 256; i++) {
for (int j = 0; j < 256; j++) {
    input[i][j] = input[j][i];
}
}   
}


 void grouped_multihead_attention_8_256_4_16_ap_fixed_16_5_(
    data_t input[8][256],
    data_t W_q[64][256],
    data_t W_k[64][256],
    data_t W_v[64][256],
    data_t output[8][64]  // for trasose purposes
)
{
    #pragma HLS inline off
    const int groups = 4;
    const int num_heads = 4;   // total number of heads (must equal DIM_OUT / HEAD_DIM)
    const int head_dim = 16;       // dimension per head
    const int heads_per_group = num_heads / groups;
    const data_t scale = (data_t)1.0 / hls::sqrt((data_t)head_dim);

    data_t Q[8][64];
    data_t K[8][64];
    data_t V[8][64];

    // Compute Q, K, V
    for (int seq = 0; seq < 8; seq++) {
        for (int dout = 0; dout < 64; dout++) {
            Q[seq][dout] = 0;
            K[seq][dout] = 0;
            V[seq][dout] = 0;
            for (int din = 0; din < 256; din++) {
                Q[seq][dout] += input[seq][din] * W_q[dout][din];
                K[seq][dout] += input[seq][din] * W_k[dout][din];
                V[seq][dout] += input[seq][din] * W_v[dout][din];
            }
        }
    }

    /*==== BEGIN OPTIONAL ROPE LOGIC ====*/
    
    /*==== END OPTIONAL ROPE LOGIC ====*/

    // Compute Attention per head in groups.
    for (int g = 0; g < groups; g++) {
        for (int h = 0; h < heads_per_group; h++) {
            int head_index = g * heads_per_group + h;
            data_t scores[8][8];

            // Scaled Dot-product: Q x K^T for head head_index
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    scores[i][j] = 0;
                    for (int d = 0; d < head_dim; d++) {
                        int idx = head_index * head_dim + d;
                        scores[i][j] += Q[i][idx] * K[j][idx];
                    }
                    scores[i][j] *= scale;
                }
            }

            // Softmax for this head.
            for (int i = 0; i < 8; i++) {
                data_t sum_exp = 0;
                data_t max_score = scores[i][0];
                for (int j = 1; j < 8; j++) {
                    if (scores[i][j] > max_score)
                        max_score = scores[i][j];
                }
                for (int j = 0; j < 8; j++) {
                    scores[i][j] = hls::exp(scores[i][j] - max_score);
                    sum_exp += scores[i][j];
                }
                for (int j = 0; j < 8; j++) {
                    scores[i][j] /= sum_exp;
                }
            }

            // Compute context: scores x V for head head_index.
            for (int i = 0; i < 8; i++) {
                for (int d = 0; d < head_dim; d++) {
                    data_t context = 0;
                    for (int j = 0; j < 8; j++) {
                        context += scores[i][j] * V[j][head_index * head_dim + d];
                    }
                    output[i][head_index * head_dim + d] = context;
                }
            }
        }
    }
}

void grouped_multihead_attention_8_256_16_16_ap_fixed_16_5_(
    data_t input[8][256],
    data_t W_q[256][256],
    data_t W_k[256][256],
    data_t W_v[256][256],
    data_t output[8][256]
)
{
    #pragma HLS inline off
    const int groups = 16; // number of groups (must divide num_heads evenly)
    const int num_heads = 16;   // total number of heads (must equal DIM_OUT / HEAD_DIM)
    const int head_dim = 16;       // dimension per head
    const int heads_per_group = num_heads / groups;
    const data_t scale = (data_t)1.0 / hls::sqrt((data_t)head_dim);

    data_t Q[8][256];
    data_t K[8][256];
    data_t V[8][256];

    // Compute Q, K, V
    for (int seq = 0; seq < 8; seq++) {
        for (int dout = 0; dout < 256; dout++) {
            Q[seq][dout] = 0;
            K[seq][dout] = 0;
            V[seq][dout] = 0;
            for (int din = 0; din < 256; din++) {
                Q[seq][dout] += input[seq][din] * W_q[dout][din];
                K[seq][dout] += input[seq][din] * W_k[dout][din];
                V[seq][dout] += input[seq][din] * W_v[dout][din];
            }
        }
    }

    /*==== BEGIN OPTIONAL ROPE LOGIC ====*/
    
    /*==== END OPTIONAL ROPE LOGIC ====*/

    // Compute Attention per head in groups.
    for (int g = 0; g < groups; g++) {
        for (int h = 0; h < heads_per_group; h++) {
            int head_index = g * heads_per_group + h;
            data_t scores[8][8];

            // Scaled Dot-product: Q x K^T for head head_index
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    scores[i][j] = 0;
                    for (int d = 0; d < head_dim; d++) {
                        int idx = head_index * head_dim + d;
                        scores[i][j] += Q[i][idx] * K[j][idx];
                    }
                    scores[i][j] *= scale;
                }
            }

            // Softmax for this head.
            for (int i = 0; i < 8; i++) {
                data_t sum_exp = 0;
                data_t max_score = scores[i][0];
                for (int j = 1; j < 8; j++) {
                    if (scores[i][j] > max_score)
                        max_score = scores[i][j];
                }
                for (int j = 0; j < 8; j++) {
                    scores[i][j] = hls::exp(scores[i][j] - max_score);
                    sum_exp += scores[i][j];
                }
                for (int j = 0; j < 8; j++) {
                    scores[i][j] /= sum_exp;
                }
            }

            // Compute context: scores x V for head head_index.
            for (int i = 0; i < 8; i++) {
                for (int d = 0; d < head_dim; d++) {
                    data_t context = 0;
                    for (int j = 0; j < 8; j++) {
                        context += scores[i][j] * V[j][head_index * head_dim + d];
                    }
                    output[i][head_index * head_dim + d] = context;
                }
            }
        }
    }
}

void grouped_mha_8_256_16_16_with_mha_4(
    data_t input[8][256],
    data_t W_q[256][256],
    data_t W_k[256][256],
    data_t W_v[256][256],
    data_t output[8][256]
)
{


    #pragma HLS inline off
    #pragma HLS allocation instances=grouped_multihead_attention_8_256_4_16_ap_fixed_16_5_ limit=1 function
    for (int i=0; i<4; i++){
        data_t split_wq[64][256];
        data_t split_wk[64][256];
        data_t split_wv[64][256];

        // Copy rows directly
        for (int r = 0; r < 64; r++) {
            for (int c = 0; c < 256; c++) {
            split_wq[r][c] = W_q[i * 64 + r][c];
            split_wk[r][c] = W_k[i * 64 + r][c];
            split_wv[r][c] = W_v[i * 64 + r][c];
            }
        }

        data_t split_out[8][64]; // temporary output for each group

        grouped_multihead_attention_8_256_4_16_ap_fixed_16_5_(input, split_wq, split_wk, split_wv, split_out); 
        // Transpose the output for concatenation
        
        for (int r=0; r<8; r++){
            for (int c=0; c<64; c++){
                output[c][i * 64 + r] = split_out[c][r];
            }
        }
    }
}

void store_8_64_ap_fixed_16_5_(data_t input[8][64], data_t output[8][64])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 8; idx0++) {
        for (int idx1 = 0; idx1 < 64; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void store_8_256_ap_fixed_16_5_(data_t input[8][256], data_t output[8][256])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 8; idx0++) {
        for (int idx1 = 0; idx1 < 256; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void top_A(data_t DRAM_attn_input[8][256], data_t DRAM_weights_q[256][256], 
    data_t DRAM_weights_k[256][256], data_t DRAM_weights_v[256][256], data_t DRAM_output[8][256])
{
    data_t BRAM_attn_input[8][256];
    data_t BRAM_weights_q[256][256];
    data_t BRAM_weights_k[256][256];
    data_t BRAM_weights_v[256][256];
    data_t BRAM_1[8][256];

    load_8_256_ap_fixed_16_5_(DRAM_attn_input, BRAM_attn_input);
    load_256_256_ap_fixed_16_5_(DRAM_weights_q, BRAM_weights_q);
    load_256_256_ap_fixed_16_5_(DRAM_weights_k, BRAM_weights_k);
    load_256_256_ap_fixed_16_5_(DRAM_weights_v, BRAM_weights_v);
    
    grouped_mha_8_256_16_16_with_mha_4(BRAM_attn_input, BRAM_weights_q, BRAM_weights_k, BRAM_weights_v, BRAM_1);
        
    store_8_256_ap_fixed_16_5_(BRAM_1, DRAM_output);
}

void top_B(data_t DRAM_attn_input[8][256], data_t DRAM_weights_q[64][256], data_t DRAM_weights_k[64][256], data_t DRAM_weights_v[64][256], data_t DRAM_output[8][64])
{
    data_t BRAM_attn_input[8][256];
    data_t BRAM_weights_q[64][256];
    data_t BRAM_weights_k[64][256];
    data_t BRAM_weights_v[64][256];
    data_t BRAM_1[8][64];

    load_8_256_ap_fixed_16_5_(DRAM_attn_input, BRAM_attn_input);
    load_64_256_ap_fixed_16_5_(DRAM_weights_q, BRAM_weights_q);
    load_64_256_ap_fixed_16_5_(DRAM_weights_k, BRAM_weights_k);
    load_64_256_ap_fixed_16_5_(DRAM_weights_v, BRAM_weights_v);

    grouped_multihead_attention_8_256_4_16_ap_fixed_16_5_(BRAM_attn_input, BRAM_weights_q, BRAM_weights_k, BRAM_weights_v, BRAM_1);
    
    store_8_64_ap_fixed_16_5_(BRAM_1, DRAM_output);
}



void top( data_t DRAM_A1[8][256], data_t DRAM_A2[256][256], data_t DRAM_A3[256][256], data_t DRAM_A4[256][256], data_t DRAM_A5[8][256],
          data_t DRAM_B1[8][256], data_t DRAM_B2[64][256], data_t DRAM_B3[64][256], data_t DRAM_B4[64][256], data_t DRAM_B5[8][64])
{
    
    #pragma HLS allocation function instances=load_8_256_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=load_256_256_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=grouped_mha_8_256_16_16_with_mha_4 limit=1
    #pragma HLS allocation function instances=grouped_multihead_attention_8_256_4_16_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=load_64_256_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=grouped_multihead_attention_8_256_4_16_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=store_8_256_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances=store_8_64_ap_fixed_16_5_ limit=1

    #pragma HLS interface m_axi port=DRAM_A1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_A2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_A3 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_A4 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_A5 offset=slave bundle=mem2
    
    #pragma HLS interface m_axi port=DRAM_B1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_B2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_B3 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_B4 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_B5 offset=slave bundle=mem2

    //TOP_A
    data_t BRAM_A1[8][256];
    data_t BRAM_A2[256][256];
    data_t BRAM_A3[256][256];
    data_t BRAM_A4[256][256];
    data_t BRAM_A5[8][256];

    load_8_256_ap_fixed_16_5_(DRAM_A1, BRAM_A1);
    load_256_256_ap_fixed_16_5_(DRAM_A2, BRAM_A2);
    load_256_256_ap_fixed_16_5_(DRAM_A3, BRAM_A3);
    load_256_256_ap_fixed_16_5_(DRAM_A4, BRAM_A4);
    // grouped_mha_8_256_16_16_with_mha_4(BRAM_A1, BRAM_A2, BRAM_A3, BRAM_A4, BRAM_A5);

    for (int i=0; i<4; i++){
        data_t split_wq[64][256];
        data_t split_wk[64][256];
        data_t split_wv[64][256];

        // Copy rows directly
        for (int r = 0; r < 64; r++) {
            for (int c = 0; c < 256; c++) {
            split_wq[r][c] = BRAM_A2[i * 64 + r][c];
            split_wk[r][c] = BRAM_A3[i * 64 + r][c];
            split_wv[r][c] = BRAM_A4[i * 64 + r][c];
            }
        }

        data_t split_out[8][64]; // temporary output for each group

        grouped_multihead_attention_8_256_4_16_ap_fixed_16_5_(BRAM_A1, split_wq, split_wk, split_wv, split_out); 
        // Transpose the output for concatenation
        
        for (int r=0; r<8; r++){
            for (int c=0; c<64; c++){
                BRAM_A5[c][i * 64 + r] = split_out[c][r];
            }
        }
    }

    store_8_256_ap_fixed_16_5_(BRAM_A5, DRAM_A5);

    //TOP_B
    data_t BRAM_B1[8][256];
    data_t BRAM_B2[64][256];
    data_t BRAM_B3[64][256];
    data_t BRAM_B4[64][256];
    data_t BRAM_B5[8][64];

    load_8_256_ap_fixed_16_5_(DRAM_B1, BRAM_B1);
    load_64_256_ap_fixed_16_5_(DRAM_B2, BRAM_B2);
    load_64_256_ap_fixed_16_5_(DRAM_B3, BRAM_B3);
    load_64_256_ap_fixed_16_5_(DRAM_B4, BRAM_B4);
    grouped_multihead_attention_8_256_4_16_ap_fixed_16_5_(BRAM_B1, BRAM_B2, BRAM_B3, BRAM_B4, BRAM_B5);
    store_8_64_ap_fixed_16_5_(BRAM_B5, DRAM_B5);
}