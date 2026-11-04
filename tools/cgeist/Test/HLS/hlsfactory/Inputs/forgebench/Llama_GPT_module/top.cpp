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

#define SEQ_LENGTH 8
#define DIM 32

data_t BRAM_attn_input[SEQ_LENGTH][DIM];
data_t BRAM_weights_q[DIM][DIM];
data_t BRAM_weights_k[DIM][DIM];
data_t BRAM_weights_v[DIM][DIM];
data_t BRAM_1[SEQ_LENGTH][DIM];
data_t BRAM_2[SEQ_LENGTH][DIM];
data_t BRAM_MLP_1[SEQ_LENGTH][4 * DIM];
data_t BRAM_MLP_2[SEQ_LENGTH][4 * DIM];
data_t BRAM_SWISH[SEQ_LENGTH][4 * DIM];
data_t BRAM_MLP_3[SEQ_LENGTH][4 * DIM];
data_t BRAM_layer_norm_weights_1[2][DIM];
data_t FF_weights_1[4 * DIM][DIM];
data_t FF_weights_2[DIM][4 * DIM];
data_t BRAM_layer_norm_weights_2[2][DIM];


void load_8_32_ap_fixed_16_5_(data_t input[SEQ_LENGTH][DIM], data_t output[SEQ_LENGTH][DIM])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < SEQ_LENGTH; idx0++) {
        for (int idx1 = 0; idx1 < DIM; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_32_32_ap_fixed_16_5_(data_t input[DIM][DIM], data_t output[DIM][DIM])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < DIM; idx0++) {
        for (int idx1 = 0; idx1 < DIM; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void load_2_32_ap_fixed_16_5_(data_t input[2][DIM], data_t output[2][DIM])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 2; idx0++) {
        for (int idx1 = 0; idx1 < DIM; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}


void load_32_128_ap_fixed_16_5_(data_t input[DIM][4 * DIM], data_t output[DIM][4 * DIM])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < DIM; idx0++) {
        for (int idx1 = 0; idx1 < 4 * DIM; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}
    
    
void load_128_32_ap_fixed_16_5_(data_t input[4 * DIM][DIM], data_t output[4 * DIM][DIM])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < 4 * DIM; idx0++) {
        for (int idx1 = 0; idx1 < DIM; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}




void layer_norm_8_32_ap_fixed_16_5_(
    data_t input[SEQ_LENGTH][DIM],
    data_t gamma[DIM],
    data_t beta[DIM],
    data_t output[SEQ_LENGTH][DIM]
)
{
    #pragma HLS inline off
    for (int i = 0; i < SEQ_LENGTH; i++) {
        // Compute mean for the i-th sequence element
        data_t sum = (data_t)0;
        for (int j = 0; j < DIM; j++) {
            sum += input[i][j];
        }
        data_t mean = sum / DIM;

        // Compute variance for the i-th sequence element
        data_t var_sum = (data_t)0;
        for (int j = 0; j < DIM; j++) {
            data_t diff = input[i][j] - mean;
            var_sum += diff * diff;
        }
        data_t variance = var_sum / DIM;

        // Normalize and scale: output = gamma * (x - mean) / sqrt(variance + epsilon) + beta
        for (int j = 0; j < DIM; j++) {
            output[i][j] = gamma[j] * ((input[i][j] - mean) / hls::sqrt(variance + (data_t)0.01)) + beta[j];
        }
    }
}


/*
 * Auto-generated Grouped Multi-head Attention (with optional inline RoPE)
 *
 * Input     : [SEQ_LENGTH][DIM]
 * W_q/k/v   : [DIM][DIM], DIM_OUT = NUM_HEADS * HEAD_DIM
 * Output    : [SEQ_LENGTH][DIM]
 *
 * Data type : ap_fixed<16, 5>
 * Num Heads : 8
 * Head Dim  : 4
 */

 
 void grouped_multihead_attention_8_32_8_4_rope_ap_fixed_16_5_(
    data_t input[SEQ_LENGTH][DIM],
    data_t W_q[DIM][DIM],
    data_t W_k[DIM][DIM],
    data_t W_v[DIM][DIM],
    data_t output[SEQ_LENGTH][DIM],
    int groups,
    int is_rope
)
{
    #pragma HLS inline off
    const int num_heads = 8;   // total number of heads (must equal DIM_OUT / HEAD_DIM)
    const int head_dim = 4;       // dimension per head
    const int heads_per_group = num_heads / groups;
    const data_t scale = (data_t)1.0 / hls::sqrt((data_t)head_dim);

    data_t Q[SEQ_LENGTH][DIM];
    data_t K[SEQ_LENGTH][DIM];
    data_t V[SEQ_LENGTH][DIM];

    // Compute Q, K, V
    for (int seq = 0; seq < SEQ_LENGTH; seq++) {
        for (int dout = 0; dout < DIM; dout++) {
            Q[seq][dout] = 0;
            K[seq][dout] = 0;
            V[seq][dout] = 0;
            for (int din = 0; din < DIM; din++) {
                Q[seq][dout] += input[seq][din] * W_q[dout][din];
                K[seq][dout] += input[seq][din] * W_k[dout][din];
                V[seq][dout] += input[seq][din] * W_v[dout][din];
            }
        }
    }

    /*==== BEGIN OPTIONAL ROPE LOGIC ====*/
    
    // Inline RoPE logic
    if (is_rope == 1)
    {
        for (int seq = 0; seq < SEQ_LENGTH; seq++) {
        for (int h = 0; h < num_heads; h++) {
            for (int d = 0; d < head_dim; d += 2) {
                int idx = h * head_dim + d;
                data_t theta = (data_t)hls::powf(10000.0f, -((float)d) / (float) head_dim);
                data_t angle = seq * theta;
                data_t cos_val = hls::cos(angle);
                data_t sin_val = hls::sin(angle);

                // Apply RoPE to Q
                data_t q0 = Q[seq][idx];
                data_t q1 = Q[seq][idx + 1];
                Q[seq][idx]     = q0 * cos_val - q1 * sin_val;
                Q[seq][idx + 1] = q0 * sin_val + q1 * cos_val;

                // Apply RoPE to K
                data_t k0 = K[seq][idx];
                data_t k1 = K[seq][idx + 1];
                K[seq][idx]     = k0 * cos_val - k1 * sin_val;
                K[seq][idx + 1] = k0 * sin_val + k1 * cos_val;
            }
        }
    }
    }
    
    /*==== END OPTIONAL ROPE LOGIC ====*/

    // Compute Attention per head in groups.
    for (int g = 0; g < groups; g++) {
        for (int h = 0; h < heads_per_group; h++) {
            int head_index = g * heads_per_group + h;
            data_t scores[SEQ_LENGTH][SEQ_LENGTH];

            // Scaled Dot-product: Q x K^T for head head_index
            for (int i = 0; i < SEQ_LENGTH; i++) {
                for (int j = 0; j < SEQ_LENGTH; j++) {
                    scores[i][j] = 0;
                    for (int d = 0; d < head_dim; d++) {
                        int idx = head_index * head_dim + d;
                        scores[i][j] += Q[i][idx] * K[j][idx];
                    }
                    scores[i][j] *= scale;
                }
            }

            // Softmax for this head.
            for (int i = 0; i < SEQ_LENGTH; i++) {
                data_t sum_exp = 0;
                data_t max_score = scores[i][0];
                for (int j = 1; j < SEQ_LENGTH; j++) {
                    if (scores[i][j] > max_score)
                        max_score = scores[i][j];
                }
                for (int j = 0; j < SEQ_LENGTH; j++) {
                    scores[i][j] = hls::exp(scores[i][j] - max_score);
                    sum_exp += scores[i][j];
                }
                for (int j = 0; j < SEQ_LENGTH; j++) {
                    scores[i][j] /= sum_exp;
                }
            }

            // Compute context: scores x V for head head_index.
            for (int i = 0; i < SEQ_LENGTH; i++) {
                for (int d = 0; d < head_dim; d++) {
                    data_t context = 0;
                    for (int j = 0; j < SEQ_LENGTH; j++) {
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
    #pragma HLS inline off
    *seed = (1103515245 * (*seed) + 12345) & 0x7fffffff;
    return *seed;
}

void dropout_8_32_ap_fixed_16_5_(
    data_t input[SEQ_LENGTH][DIM],
    data_t output[SEQ_LENGTH][DIM],
    data_t dropout_prob,
    unsigned int seed
)
{
    #pragma HLS inline off
    for (int i = 0; i < SEQ_LENGTH; i++) {
        for (int j = 0; j < DIM; j++) {
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
    data_t in1[SEQ_LENGTH][DIM],
    data_t in2[SEQ_LENGTH][DIM],
    data_t out[SEQ_LENGTH][DIM]
)
{
    #pragma HLS inline off
    
    for (int i = 0; i < SEQ_LENGTH; i++) {
        for (int j = 0; j < DIM; j++) {
            out[i][j] = in1[i][j] + in2[i][j];
        }
    }
    
}






void gelu_8_128_ap_fixed_16_5_(
    data_t input[SEQ_LENGTH][4 * DIM],
    data_t output[SEQ_LENGTH][4 * DIM]
)
{
    #pragma HLS inline off
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const data_t sqrt_2_over_pi = hls::sqrt((data_t)2/(data_t)3.141592653589793);
    for (int i = 0; i < SEQ_LENGTH; i++) {
        for (int j = 0; j < 4 * DIM; j++) {
            data_t x = input[i][j];
            data_t x_cube = x * x * x;
            data_t tanh_arg = sqrt_2_over_pi * (x + (data_t)0.044715 * x_cube);
            output[i][j] = (data_t)0.5 * x * (1 + hls::tanh(tanh_arg));
        }
    }
}


void matmul_8_32_128_ap_fixed_16_5_(
    data_t input[SEQ_LENGTH][DIM],
    data_t weights[4 * DIM][DIM],
    data_t output[SEQ_LENGTH][4 * DIM]
)
{
    #pragma HLS inline off
    // Initialize output to ((data_t)0)
    for (int i = 0; i < SEQ_LENGTH; i++) {
        for (int j = 0; j < 4 * DIM; j++) {
            output[i][j] = ((data_t)0);
        }
    }

    // Matrix multiplication
    for (int i = 0; i < SEQ_LENGTH; i++) {
        for (int k = 0; k < DIM; k++) {
            for (int j = 0; j < 4 * DIM; j++) {
                output[i][j] += input[i][k] * weights[j][k];
            }
        }
    }
}



void matmul_8_128_32_ap_fixed_16_5_(
    data_t input[SEQ_LENGTH][4 * DIM],
    data_t weights[DIM][4 * DIM],
    data_t output[SEQ_LENGTH][DIM]
)
{
    #pragma HLS inline off
    // Initialize output to ((data_t)0)
    for (int i = 0; i < SEQ_LENGTH; i++) {
        for (int j = 0; j < 4 * DIM; j++) {
            output[i][j] = ((data_t)0);
        }
    }

    // Matrix multiplication
    for (int i = 0; i < SEQ_LENGTH; i++) {
        for (int k = 0; k < 4 * DIM; k++) {
            for (int j = 0; j < DIM; j++) {
                output[i][j] += input[i][k] * weights[j][k];
            }
        }
    }
}





void store_8_32_ap_fixed_16_5_(data_t input[SEQ_LENGTH][DIM], data_t output[SEQ_LENGTH][DIM])
{
    #pragma HLS inline off
    for (int idx0 = 0; idx0 < SEQ_LENGTH; idx0++) {
        for (int idx1 = 0; idx1 < DIM; idx1++) {
            output[idx0][idx1] = input[idx0][idx1];
        }
    }
}

void rms_norm_8_32_ap_fixed_16_5_(
    data_t input[SEQ_LENGTH][DIM],
    data_t gamma[DIM],
    data_t output[SEQ_LENGTH][DIM]
)
{
    #pragma HLS inline off
    for (int i = 0; i < SEQ_LENGTH; i++) {
        data_t sum_sq = (data_t)0;
        for (int j = 0; j < DIM; j++) {
            sum_sq += input[i][j] * input[i][j];
        }
        data_t rms = hls::sqrt(sum_sq / (data_t) DIM + (data_t) 0.01);
        for (int j = 0; j < DIM; j++) {
            output[i][j] = gamma[j] * input[i][j] / rms;
        }
    }
}


void swish_8_128_ap_fixed_16_5_(
    data_t input[SEQ_LENGTH][4 * DIM],
    data_t output[SEQ_LENGTH][4 * DIM]
)
{
    #pragma HLS inline off
    // Swish: x * sigmoid(x)
    for (int i = 0; i < SEQ_LENGTH; i++) {
        for (int j = 0; j < 4 * DIM; j++) {
            data_t sig = (data_t)1 / ((data_t)1 + hls::exp(-input[i][j]));
            output[i][j] = input[i][j] * sig;
        }
    }
}


void elementwise_mult_8_128_ap_fixed_16_5_(
    data_t in1[SEQ_LENGTH][4 * DIM],
    data_t in2[SEQ_LENGTH][4 * DIM],
    data_t out[SEQ_LENGTH][4 * DIM]
)
{
    #pragma HLS inline off
    
    for (int i = 0; i < SEQ_LENGTH; i++) {
        for (int j = 0; j < 4 * DIM; j++) {
            out[i][j] = in1[i][j] * in2[i][j];
        }
    }
    
}

void matmul_8_32_128_bias_ap_fixed_16_5_(
    data_t input[SEQ_LENGTH][DIM],
    data_t weights[4 * DIM][DIM],
    data_t bias[4 * DIM],
    data_t output[SEQ_LENGTH][4 * DIM]
)
{
    #pragma HLS inline off
    // Initialize output to bias[j]
    for (int i = 0; i < SEQ_LENGTH; i++) {
        for (int j = 0; j < 4 * DIM; j++) {
            output[i][j] = bias[j];
        }
    }

    // Matrix multiplication
    for (int i = 0; i < SEQ_LENGTH; i++) {
        for (int k = 0; k < DIM; k++) {
            for (int j = 0; j < 4 * DIM; j++) {
                output[i][j] += input[i][k] * weights[j][k];
            }
        }
    }
}


void gpt(data_t DRAM_attn_input[SEQ_LENGTH][DIM], data_t DRAM_weights_q[DIM][DIM], data_t DRAM_weights_k[DIM][DIM], data_t DRAM_weights_v[DIM][DIM], data_t DRAM_layer_norm_weights_1[2][DIM], data_t DRAM_FF_weights_1[4 * DIM][DIM], data_t DRAM_FF_weights_2[DIM][4 * DIM], data_t DRAM_layer_norm_weights_2[2][DIM], data_t DRAM_output[SEQ_LENGTH][DIM])
{
    #pragma HLS inline off
    
    load_8_32_ap_fixed_16_5_(DRAM_attn_input, BRAM_attn_input);
    load_32_32_ap_fixed_16_5_(DRAM_weights_q, BRAM_weights_q);
    load_32_32_ap_fixed_16_5_(DRAM_weights_k, BRAM_weights_k);
    load_32_32_ap_fixed_16_5_(DRAM_weights_v, BRAM_weights_v);
    load_2_32_ap_fixed_16_5_(DRAM_layer_norm_weights_1, BRAM_layer_norm_weights_1);
    load_128_32_ap_fixed_16_5_(DRAM_FF_weights_1, FF_weights_1);
    load_32_128_ap_fixed_16_5_(DRAM_FF_weights_2, FF_weights_2);
    load_2_32_ap_fixed_16_5_(DRAM_layer_norm_weights_2, BRAM_layer_norm_weights_2);
    layer_norm_8_32_ap_fixed_16_5_(BRAM_attn_input, BRAM_layer_norm_weights_1[0], BRAM_layer_norm_weights_1[1], BRAM_2);
    grouped_multihead_attention_8_32_8_4_rope_ap_fixed_16_5_(BRAM_2, BRAM_weights_q, BRAM_weights_k, BRAM_weights_v, BRAM_1, 8, 0);
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

void llama(data_t DRAM_attn_input[SEQ_LENGTH][DIM], data_t DRAM_weights_q[DIM][DIM], data_t DRAM_weights_k[DIM][DIM], data_t DRAM_weights_v[DIM][DIM], data_t DRAM_rms_norm_weights_1[2][DIM], data_t DRAM_FF_weights_1[4 * DIM][DIM], data_t DRAM_FF_weights_2[DIM][4 * DIM], data_t DRAM_rms_norm_weights_2[2][DIM], data_t DRAM_output[SEQ_LENGTH][DIM])
{
    #pragma HLS inline off
    
    load_8_32_ap_fixed_16_5_(DRAM_attn_input, BRAM_attn_input);
    load_32_32_ap_fixed_16_5_(DRAM_weights_q, BRAM_weights_q);
    load_32_32_ap_fixed_16_5_(DRAM_weights_k, BRAM_weights_k);
    load_32_32_ap_fixed_16_5_(DRAM_weights_v, BRAM_weights_v);
    load_2_32_ap_fixed_16_5_(DRAM_rms_norm_weights_1, BRAM_layer_norm_weights_1);
    load_128_32_ap_fixed_16_5_(DRAM_FF_weights_1, FF_weights_1);
    load_32_128_ap_fixed_16_5_(DRAM_FF_weights_2, FF_weights_2);
    load_2_32_ap_fixed_16_5_(DRAM_rms_norm_weights_2, BRAM_layer_norm_weights_2);
    rms_norm_8_32_ap_fixed_16_5_(BRAM_attn_input, BRAM_layer_norm_weights_1[0], BRAM_2);
    grouped_multihead_attention_8_32_8_4_rope_ap_fixed_16_5_(BRAM_2, BRAM_weights_q, BRAM_weights_k, BRAM_weights_v, BRAM_1, 4, 1);
    dropout_8_32_ap_fixed_16_5_(BRAM_1, BRAM_2, 0.5, 47);
    matrix_add_8_32_ap_fixed_16_5_(BRAM_attn_input, BRAM_2, BRAM_1);
    rms_norm_8_32_ap_fixed_16_5_(BRAM_1, BRAM_layer_norm_weights_2[0], BRAM_2);
    matmul_8_32_128_ap_fixed_16_5_(BRAM_2, FF_weights_1, BRAM_MLP_1);
    swish_8_128_ap_fixed_16_5_(BRAM_MLP_1, BRAM_SWISH);
    elementwise_mult_8_128_ap_fixed_16_5_(BRAM_SWISH, BRAM_MLP_1, BRAM_MLP_2);
    matmul_8_128_32_ap_fixed_16_5_(BRAM_MLP_2, FF_weights_2, BRAM_1);
    dropout_8_32_ap_fixed_16_5_(BRAM_1, BRAM_2, 0.5, 47);
    matrix_add_8_32_ap_fixed_16_5_(BRAM_2, BRAM_1, BRAM_2);
    store_8_32_ap_fixed_16_5_(BRAM_2, DRAM_output);
}

void top(data_t DRAM_attn_input_gpt[SEQ_LENGTH][DIM], data_t DRAM_weights_q_gpt[DIM][DIM], data_t DRAM_weights_k_gpt[DIM][DIM], data_t DRAM_weights_v_gpt[DIM][DIM], data_t DRAM_layer_norm_weights_1_gpt[2][DIM], data_t DRAM_FF_weights_1_gpt[4 * DIM][DIM], data_t DRAM_FF_weights_2_gpt[DIM][4 * DIM], data_t DRAM_layer_norm_weights_2_gpt[2][DIM], data_t DRAM_output_gpt[SEQ_LENGTH][DIM],
        data_t DRAM_attn_input_llama[SEQ_LENGTH][DIM], data_t DRAM_weights_q_llama[DIM][DIM], data_t DRAM_weights_k_llama[DIM][DIM], data_t DRAM_weights_v_llama[DIM][DIM], data_t DRAM_rms_norm_weights_1_llama[2][DIM], data_t DRAM_FF_weights_1_llama[4 * DIM][DIM], data_t DRAM_FF_weights_2_llama[DIM][4 * DIM], data_t DRAM_rms_norm_weights_2_llama[2][DIM], data_t DRAM_output_llama[SEQ_LENGTH][DIM])
{
    #pragma HLS interface m_axi port=DRAM_attn_input_gpt offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_weights_q_gpt offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_weights_k_gpt offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_weights_v_gpt offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_layer_norm_weights_1_gpt offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_FF_weights_1_gpt offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_FF_weights_2_gpt offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_layer_norm_weights_2_gpt offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_output_gpt offset=slave bundle=mem2

    #pragma HLS interface m_axi port=DRAM_attn_input_llama offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_weights_q_llama offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_weights_k_llama offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_weights_v_llama offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_rms_norm_weights_1_llama offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_FF_weights_1_llama offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_FF_weights_2_llama offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_rms_norm_weights_2_llama offset=slave bundle=mem1
    #pragma HLS interface m_axi port=DRAM_output_llama offset=slave bundle=mem2

    #pragma HLS allocation function instances= load_8_32_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances= load_32_32_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances= load_2_32_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances= load_128_32_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances= load_32_128_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances= layer_norm_8_32_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances= grouped_multihead_attention_8_32_8_4_rope_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances= dropout_8_32_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances= matrix_add_8_32_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances= matmul_8_32_128_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances= gelu_8_128_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances= matmul_8_128_32_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances= store_8_32_ap_fixed_16_5_ limit=1


    #pragma HLS allocation function instances= rms_norm_8_32_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances= swish_8_128_ap_fixed_16_5_ limit=1
    #pragma HLS allocation function instances= elementwise_mult_8_128_ap_fixed_16_5_ limit=1


    //gpt
    load_8_32_ap_fixed_16_5_(DRAM_attn_input_gpt, BRAM_attn_input);
    load_32_32_ap_fixed_16_5_(DRAM_weights_q_gpt, BRAM_weights_q);
    load_32_32_ap_fixed_16_5_(DRAM_weights_k_gpt, BRAM_weights_k);
    load_32_32_ap_fixed_16_5_(DRAM_weights_v_gpt, BRAM_weights_v);
    load_2_32_ap_fixed_16_5_(DRAM_layer_norm_weights_1_gpt, BRAM_layer_norm_weights_1);
    load_128_32_ap_fixed_16_5_(DRAM_FF_weights_1_gpt, FF_weights_1);
    load_32_128_ap_fixed_16_5_(DRAM_FF_weights_2_gpt, FF_weights_2);
    load_2_32_ap_fixed_16_5_(DRAM_layer_norm_weights_2_gpt, BRAM_layer_norm_weights_2);
    layer_norm_8_32_ap_fixed_16_5_(BRAM_attn_input, BRAM_layer_norm_weights_1[0], BRAM_layer_norm_weights_1[1], BRAM_2);
    grouped_multihead_attention_8_32_8_4_rope_ap_fixed_16_5_(BRAM_2, BRAM_weights_q, BRAM_weights_k, BRAM_weights_v, BRAM_1, 8, 0);
    dropout_8_32_ap_fixed_16_5_(BRAM_1, BRAM_2, 0.5, 47);
    matrix_add_8_32_ap_fixed_16_5_(BRAM_attn_input, BRAM_2, BRAM_1);
    layer_norm_8_32_ap_fixed_16_5_(BRAM_1, BRAM_layer_norm_weights_2[0], BRAM_layer_norm_weights_2[1], BRAM_2);
    matmul_8_32_128_ap_fixed_16_5_(BRAM_2, FF_weights_1, BRAM_MLP_1);
    gelu_8_128_ap_fixed_16_5_(BRAM_MLP_1, BRAM_MLP_2);
    matmul_8_128_32_ap_fixed_16_5_(BRAM_MLP_2, FF_weights_2, BRAM_1);
    dropout_8_32_ap_fixed_16_5_(BRAM_1, BRAM_2, 0.5, 47);
    matrix_add_8_32_ap_fixed_16_5_(BRAM_2, BRAM_1, BRAM_2);
    store_8_32_ap_fixed_16_5_(BRAM_2, DRAM_output_gpt);

    //llama
    load_8_32_ap_fixed_16_5_(DRAM_attn_input_llama, BRAM_attn_input);
    load_32_32_ap_fixed_16_5_(DRAM_weights_q_llama, BRAM_weights_q);
    load_32_32_ap_fixed_16_5_(DRAM_weights_k_llama, BRAM_weights_k);
    load_32_32_ap_fixed_16_5_(DRAM_weights_v_llama, BRAM_weights_v);
    load_2_32_ap_fixed_16_5_(DRAM_rms_norm_weights_1_llama, BRAM_layer_norm_weights_1);
    load_128_32_ap_fixed_16_5_(DRAM_FF_weights_1_llama, FF_weights_1);
    load_32_128_ap_fixed_16_5_(DRAM_FF_weights_2_llama, FF_weights_2);
    load_2_32_ap_fixed_16_5_(DRAM_rms_norm_weights_2_llama, BRAM_layer_norm_weights_2);
    rms_norm_8_32_ap_fixed_16_5_(BRAM_attn_input, BRAM_layer_norm_weights_1[0], BRAM_2);
    grouped_multihead_attention_8_32_8_4_rope_ap_fixed_16_5_(BRAM_2, BRAM_weights_q, BRAM_weights_k, BRAM_weights_v, BRAM_1, 4, 1);
    dropout_8_32_ap_fixed_16_5_(BRAM_1, BRAM_2, 0.5, 47);
    matrix_add_8_32_ap_fixed_16_5_(BRAM_attn_input, BRAM_2, BRAM_1);
    rms_norm_8_32_ap_fixed_16_5_(BRAM_1, BRAM_layer_norm_weights_1[0], BRAM_2);
    matmul_8_32_128_ap_fixed_16_5_(BRAM_2, FF_weights_1, BRAM_MLP_1);
    swish_8_128_ap_fixed_16_5_(BRAM_MLP_1, BRAM_SWISH);
    elementwise_mult_8_128_ap_fixed_16_5_(BRAM_SWISH, BRAM_MLP_1, BRAM_MLP_2);
    matmul_8_128_32_ap_fixed_16_5_(BRAM_MLP_2, FF_weights_2, BRAM_1);
    dropout_8_32_ap_fixed_16_5_(BRAM_1, BRAM_2, 0.5, 47);
    matrix_add_8_32_ap_fixed_16_5_(BRAM_2, BRAM_1, BRAM_2);
    store_8_32_ap_fixed_16_5_(BRAM_2, DRAM_output_llama);

    // gpt(DRAM_attn_input_gpt, DRAM_weights_q_gpt, DRAM_weights_k_gpt, DRAM_weights_v_gpt, DRAM_layer_norm_weights_1_gpt, DRAM_FF_weights_1_gpt, DRAM_FF_weights_2_gpt, DRAM_layer_norm_weights_2_gpt, DRAM_output_gpt);
    // llama(DRAM_attn_input_llama, DRAM_weights_q_llama, DRAM_weights_k_llama, DRAM_weights_v_llama, DRAM_rms_norm_weights_1_llama, DRAM_FF_weights_1_llama, DRAM_FF_weights_2_llama, DRAM_rms_norm_weights_2_llama, DRAM_output_llama);

}






