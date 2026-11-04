
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

data_t BRAM_attn_input[8][32];
data_t BRAM_weights_q[32][32];
data_t BRAM_weights_k[32][32];
data_t BRAM_weights_v[32][32];
data_t BRAM_1[8][32];
data_t BRAM_2[8][32];
data_t BRAM_MLP_1[8][128];
data_t BRAM_MLP_2[8][128];
data_t BRAM_layer_norm_weights_1[2][32];
data_t FF_weights_1[128][32];
data_t FF_weights_2[32][128];
data_t BRAM_layer_norm_weights_2[2][32];

void load_8_32_ap_fixed_16_5_(data_t input[8][32], data_t output[8][32])
{
    for (int idx0 = 0; idx0 < 8; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_32_32_ap_fixed_16_5_(data_t input[32][32], data_t output[32][32])
{
    for (int idx0 = 0; idx0 < 32; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_2_32_ap_fixed_16_5_(data_t input[2][32], data_t output[2][32])
{
    for (int idx0 = 0; idx0 < 2; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_128_32_ap_fixed_16_5_(data_t input[128][32], data_t output[128][32])
{
    for (int idx0 = 0; idx0 < 128; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_32_128_ap_fixed_16_5_(data_t input[32][128], data_t output[32][128])
{
    for (int idx0 = 0; idx0 < 32; idx0++) {
        for (int idx1 = 0; idx1 < 128; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}



void layer_norm_8_32_ap_fixed_16_5_(
    data_t input[8][32],
    data_t gamma[32],
    data_t beta[32],
    data_t output[8][32]
)
{
    for (int i = 0; i < 8; i++) {
        // Compute mean for the i-th sequence element
        data_t sum = (data_t)0;
        for (int j = 0; j < 32; j++) {
            sum += input[i][j];
        }
        data_t mean = sum / 32;

        // Compute variance for the i-th sequence element
        data_t var_sum = (data_t)0;
        for (int j = 0; j < 32; j++) {
            data_t diff = input[i][j] - mean;
            var_sum += diff * diff;
        }
        data_t variance = var_sum / 32;

        // Normalize and scale: output = gamma * (x - mean) / sqrt(variance + epsilon) + beta
        for (int j = 0; j < 32; j++) {
            output[i][j] = gamma[j] * ((input[i][j] - mean) / hls::sqrt(variance + (data_t)0.01)) + beta[j];
        }
    }
}


/*
 * Auto-generated Grouped Multi-head Attention (with optional inline RoPE)
 *
 * Input     : [8][32]
 * W_q/k/v   : [32][32], DIM_OUT = NUM_HEADS * HEAD_DIM
 * Output    : [8][32]
 *
 * Data type : ap_fixed<16, 5>
 * Num Heads : 8
 * Head Dim  : 4
 */

 void grouped_multihead_attention_8_32_8_4_ap_fixed_16_5_(
    data_t input[8][32],
    data_t W_q[32][32],
    data_t W_k[32][32],
    data_t W_v[32][32],
    data_t output[8][32],
    int groups
)
{
    const int num_heads = 8;   // total number of heads (must equal DIM_OUT / HEAD_DIM)
    const int head_dim = 4;       // dimension per head
    const int heads_per_group = num_heads / groups;
    const data_t scale = (data_t)1.0 / hls::sqrt((data_t)head_dim);

    data_t Q[8][32];
    data_t K[8][32];
    data_t V[8][32];

    // Compute Q, K, V
    for (int seq = 0; seq < 8; seq++) {
        for (int dout = 0; dout < 32; dout++) {
            Q[seq][dout] = 0;
            K[seq][dout] = 0;
            V[seq][dout] = 0;
            for (int din = 0; din < 32; din++) {
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


// Simple LCG for pseudo-random numbers.
unsigned int lcg_rand(unsigned int *seed) {
    *seed = (1103515245 * (*seed) + 12345) & 0x7fffffff;
    return *seed;
}

void dropout_8_32_ap_fixed_16_5_(
    data_t input[8][32],
    data_t output[8][32],
    data_t dropout_prob,
    unsigned int seed
)
{
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 32; j++) {
            // Generate a random value between 0 and 1.
            unsigned int r = lcg_rand(&seed);
            data_t rand_val = (data_t)r / (data_t)2147483647;
            // Apply dropout: if random value is below dropout_prob, output is 0;
            // otherwise, scale the input.
            if (rand_val < dropout_prob) {
                output[i][j] = (data_t)0;
            } else {
                output[i][j] = input[i][j] / (1 - dropout_prob);
            }
        }
    }
}



void matrix_add_8_32_ap_fixed_16_5_(
    data_t in1[8][32],
    data_t in2[8][32],
    data_t out[8][32]
)
{
    
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 32; j++) {
            out[i][j] = in1[i][j] + in2[i][j];
        }
    }
    
}









void matmul_8_32_128_ap_fixed_16_5_(
    data_t input[8][32],
    data_t weights[128][32],
    data_t output[8][128]
)
{
    // Initialize output to ((data_t)0)
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 128; j++) {
            output[i][j] = ((data_t)0);
        }
    }

    // Matrix multiplication
    for (int i = 0; i < 8; i++) {
        for (int k = 0; k < 32; k++) {
            for (int j = 0; j < 128; j++) {
                output[i][j] += input[i][k] * weights[j][k];
            }
        }
    }
}




void gelu_8_128_ap_fixed_16_5_(
    data_t input[8][128],
    data_t output[8][128]
)
{
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const data_t sqrt_2_over_pi = hls::sqrt((data_t)2/(data_t)3.141592653589793);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 128; j++) {
            data_t x = input[i][j];
            data_t x_cube = x * x * x;
            data_t tanh_arg = sqrt_2_over_pi * (x + (data_t)0.044715 * x_cube);
            output[i][j] = (data_t)0.5 * x * (1 + hls::tanh(tanh_arg));
        }
    }
}



void matmul_8_128_32_ap_fixed_16_5_(
    data_t input[8][128],
    data_t weights[32][128],
    data_t output[8][32]
)
{
    // Initialize output to ((data_t)0)
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 32; j++) {
            output[i][j] = ((data_t)0);
        }
    }

    // Matrix multiplication
    for (int i = 0; i < 8; i++) {
        for (int k = 0; k < 128; k++) {
            for (int j = 0; j < 32; j++) {
                output[i][j] += input[i][k] * weights[j][k];
            }
        }
    }
}


void store_8_32_ap_fixed_16_5_(data_t input[8][32], data_t output[8][32])
{
    for (int idx0 = 0; idx0 < 8; idx0++) {
        for (int idx1 = 0; idx1 < 32; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void top(data_t DRAM_attn_input[8][32], data_t DRAM_weights_q[32][32], data_t DRAM_weights_k[32][32], data_t DRAM_weights_v[32][32], data_t DRAM_layer_norm_weights_1[2][32], data_t DRAM_FF_weights_1[128][32], data_t DRAM_FF_weights_2[32][128], data_t DRAM_layer_norm_weights_2[2][32], data_t DRAM_output[8][32])
{
    #pragma HLS interface m_axi port=DRAM_attn_input offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_weights_q offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_weights_k offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_weights_v offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_layer_norm_weights_1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_FF_weights_1 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_FF_weights_2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_layer_norm_weights_2 offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_output offset=slave bundle=mem2

    load_8_32_ap_fixed_16_5_(DRAM_attn_input, BRAM_attn_input);
    load_32_32_ap_fixed_16_5_(DRAM_weights_q, BRAM_weights_q);
    load_32_32_ap_fixed_16_5_(DRAM_weights_k, BRAM_weights_k);
    load_32_32_ap_fixed_16_5_(DRAM_weights_v, BRAM_weights_v);
    load_2_32_ap_fixed_16_5_(DRAM_layer_norm_weights_1, BRAM_layer_norm_weights_1);
    load_128_32_ap_fixed_16_5_(DRAM_FF_weights_1, FF_weights_1);
    load_32_128_ap_fixed_16_5_(DRAM_FF_weights_2, FF_weights_2);
    load_2_32_ap_fixed_16_5_(DRAM_layer_norm_weights_2, BRAM_layer_norm_weights_2);
    layer_norm_8_32_ap_fixed_16_5_(BRAM_attn_input, BRAM_layer_norm_weights_1[0], BRAM_layer_norm_weights_1[1], BRAM_2);
    grouped_multihead_attention_8_32_8_4_ap_fixed_16_5_(BRAM_2, BRAM_weights_q, BRAM_weights_k, BRAM_weights_v, BRAM_1, 8);
    dropout_8_32_ap_fixed_16_5_(BRAM_1, BRAM_2, 0.5, 47);
    matrix_add_8_32_ap_fixed_16_5_(BRAM_attn_input, BRAM_2, BRAM_1);
    layer_norm_8_32_ap_fixed_16_5_(BRAM_1, BRAM_layer_norm_weights_2[0], BRAM_layer_norm_weights_2[1], BRAM_2);
    matmul_8_32_128_ap_fixed_16_5_(BRAM_2, FF_weights_1, BRAM_MLP_1);
    gelu_8_128_ap_fixed_16_5_(BRAM_MLP_1, BRAM_MLP_2);
    matmul_8_128_32_ap_fixed_16_5_(BRAM_MLP_2, FF_weights_2, BRAM_1);
    dropout_8_32_ap_fixed_16_5_(BRAM_1, BRAM_2, 0.5, 47);
    matrix_add_8_32_ap_fixed_16_5_(BRAM_2, BRAM_1, BRAM_2);
    store_8_32_ap_fixed_16_5_(BRAM_2, DRAM_output);
}