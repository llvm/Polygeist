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
#define NUM_HEAD 8
#define HEAD_DIM 4

data_t embedding_buffer_1[SEQ_LENGTH][DIM];

data_t Q_buffer[SEQ_LENGTH][DIM];
data_t Q_buffer_rope[SEQ_LENGTH][DIM];
data_t Q_weight_buffer[DIM][DIM];

data_t K_buffer[SEQ_LENGTH][DIM];
data_t K_buffer_rope[SEQ_LENGTH][DIM];
data_t K_weight_buffer[DIM][DIM];

data_t V_buffer[SEQ_LENGTH][DIM];
data_t V_weight_buffer[DIM][DIM];

data_t scores[SEQ_LENGTH][SEQ_LENGTH];
data_t scores_2[SEQ_LENGTH][SEQ_LENGTH];

void matmul_large (data_t input_1[SEQ_LENGTH][DIM], data_t input_2[SEQ_LENGTH][DIM], data_t output[SEQ_LENGTH][DIM])
{
    #pragma HLS inline off
    for (int seq = 0; seq < SEQ_LENGTH; seq++) {
        for (int dout = 0; dout < DIM; dout++) {
            output[seq][dout] = 0;
            for (int din = 0; din < DIM; din++) {
                output[seq][dout] += input_1[seq][din] * input_2[dout][din];
            }
        }
    }
}

void matmul_small (data_t input_1[SEQ_LENGTH][HEAD_DIM], data_t input_2[SEQ_LENGTH][HEAD_DIM], data_t output[SEQ_LENGTH][HEAD_DIM])
{
    #pragma HLS inline off
    for (int seq = 0; seq < SEQ_LENGTH; seq++) {
        for (int dout = 0; dout < HEAD_DIM; dout++) {
            for (int din = 0; din < SEQ_LENGTH; din++) {
                output[seq][dout] += input_1[seq][din] * input_2[dout][din];
            }
        }
    }
}

void matmul_small_1 (data_t Q[SEQ_LENGTH][DIM], data_t K[SEQ_LENGTH][DIM], data_t scores[SEQ_LENGTH][SEQ_LENGTH], int head_index)
{
    #pragma HLS inline off
    // Scaled Dot-product: Q x K^T for head head_index
    for (int i = 0; i < SEQ_LENGTH; i++) {
        for (int j = 0; j < SEQ_LENGTH; j++) {
            scores[i][j] = 0;
            for (int d = 0; d < HEAD_DIM; d++) {
                int idx = head_index * HEAD_DIM + d;
                scores[i][j] += Q[i][idx] * K[j][idx];
            }
        }
    }
}

void matmul_small_2 (data_t scores[SEQ_LENGTH][SEQ_LENGTH], data_t V[SEQ_LENGTH][DIM], data_t output[SEQ_LENGTH][DIM], int head_index)
{
    #pragma HLS inline off
    // Compute context: scores x V for head head_index.
    for (int i = 0; i < SEQ_LENGTH; i++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            data_t context = 0;
            for (int j = 0; j < SEQ_LENGTH; j++) {
                context += scores[i][j] * V[j][head_index * HEAD_DIM + d];
            }
            output[i][head_index * HEAD_DIM + d] = context;
        }
    }
}

void rope(
    data_t input[SEQ_LENGTH][DIM],
    data_t output[SEQ_LENGTH][DIM]
)
{
    // Apply RoPE to each sequence element.
    #pragma HLS inline off

    for (int seq = 0; seq < SEQ_LENGTH; seq++) {
        for (int h = 0; h < NUM_HEAD; h++) {
            for (int d = 0; d < HEAD_DIM; d += 2) {
                int idx = h * HEAD_DIM + d;
                data_t theta = (data_t)hls::powf(10000.0f, -((float)d) / (float) HEAD_DIM);
                data_t angle = seq * theta;
                data_t cos_val = hls::cos(angle);
                data_t sin_val = hls::sin(angle);

                // Apply RoPE to Q
                data_t q0 = input[seq][idx];
                data_t q1 = input[seq][idx + 1];
                output[seq][idx]     = q0 * cos_val - q1 * sin_val;
                output[seq][idx + 1] = q0 * sin_val + q1 * cos_val;
            }
        }
    }
}

void softmax(
    data_t input[SEQ_LENGTH][SEQ_LENGTH],
    data_t output[SEQ_LENGTH][SEQ_LENGTH]
)
{
    // Compute softmax along the hidden dimension for each row.
    #pragma HLS inline off
    for (int i = 0; i < SEQ_LENGTH; i++) {
        data_t sum = 0;
        for (int j = 0; j < DIM; j++) {
            output[i][j] = hls::exp(input[i][j]);
            sum += output[i][j];
        }
        for (int j = 0; j < DIM; j++) {
            output[i][j] /= sum;
        }
    }
}

void calculate_scale(data_t input[SEQ_LENGTH][SEQ_LENGTH], data_t output [SEQ_LENGTH][SEQ_LENGTH])
{
    #pragma HLS inline off
    data_t scale = (data_t)1.0 / hls::sqrt((data_t)HEAD_DIM);
    for (int i = 0; i < SEQ_LENGTH; i++)
    {
        for (int j = 0; j < SEQ_LENGTH; j++)
        {
            output[i][j] = input[i][j] * scale;
        }
    }

}

void load_embedding(data_t input_dram[SEQ_LENGTH][DIM], data_t input_buffer[SEQ_LENGTH][DIM])
{
    #pragma HLS inline off
    for (int i = 0; i < SEQ_LENGTH; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            input_buffer[i][j] = input_dram[i][j];
        }
    }
}

void store_embedding(data_t output_buffer[SEQ_LENGTH][DIM], data_t output_dram[SEQ_LENGTH][DIM])
{
    #pragma HLS inline off
    for (int i = 0; i < SEQ_LENGTH; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            output_dram[i][j] = output_buffer[i][j];
        }
    }
}

void load_weights(data_t input_dram[DIM][DIM], data_t input_buffer[DIM][DIM])
{
    #pragma HLS inline off
    for (int i = 0; i < DIM; i++)
    {
        for (int j = 0; j < DIM; j++)
        {
            input_buffer[i][j] = input_dram[i][j];
        }
    }
}

void top_B(data_t input_dram[SEQ_LENGTH][DIM], data_t Q_weight_dram[DIM][DIM], data_t K_weight_dram[DIM][DIM], data_t V_weight_dram[DIM][DIM], data_t output_dram[SEQ_LENGTH][DIM])
{
    #pragma HLS inline off
    const int groups = 4;
    const int heads_per_group = NUM_HEAD / groups;
    load_embedding(input_dram, embedding_buffer_1);
    load_weights(Q_weight_dram, Q_weight_buffer);
    load_weights(K_weight_dram, K_weight_buffer);
    load_weights(V_weight_dram, V_weight_buffer);
    matmul_large(embedding_buffer_1, Q_weight_buffer, Q_buffer);
    matmul_large(embedding_buffer_1, K_weight_buffer, K_buffer);
    matmul_large(embedding_buffer_1, V_weight_buffer, V_buffer);

    rope(Q_buffer, Q_buffer_rope);
    rope(K_buffer, K_buffer_rope);
    for (int g = 0; g < groups; g++) {
        for (int h = 0; h < heads_per_group; h++) {
            int head_index = g * heads_per_group + h;

            matmul_small_1(Q_buffer_rope, K_buffer_rope, scores, head_index);
            calculate_scale(scores, scores_2);
            softmax(scores_2, scores);
            matmul_small_2(scores, V_buffer, embedding_buffer_1, head_index);
        }
    }

    store_embedding(embedding_buffer_1, output_dram);

}

void top_A(data_t input_dram[SEQ_LENGTH][DIM], data_t Q_weight_dram[DIM][DIM], data_t K_weight_dram[DIM][DIM], data_t V_weight_dram[DIM][DIM], data_t output_dram[SEQ_LENGTH][DIM])
{
    #pragma HLS inline off
    const int groups = NUM_HEAD;
    const int heads_per_group = NUM_HEAD / groups;
    load_embedding(input_dram, embedding_buffer_1);
    load_weights(Q_weight_dram, Q_weight_buffer);
    load_weights(K_weight_dram, K_weight_buffer);
    load_weights(V_weight_dram, V_weight_buffer);
    matmul_large(embedding_buffer_1, Q_weight_buffer, Q_buffer);
    matmul_large(embedding_buffer_1, K_weight_buffer, K_buffer);
    matmul_large(embedding_buffer_1, V_weight_buffer, V_buffer);



    for (int g = 0; g < groups; g++) {
        for (int h = 0; h < heads_per_group; h++) {
            int head_index = g * heads_per_group + h;

            matmul_small_1(Q_buffer, K_buffer, scores, head_index);
            calculate_scale(scores, scores_2);
            softmax(scores_2, scores);
            matmul_small_2(scores, V_buffer, embedding_buffer_1, head_index);
        }
    }

    store_embedding(embedding_buffer_1, output_dram);

}


void top(data_t input_dram_A[SEQ_LENGTH][DIM], data_t Q_weight_dram_A[DIM][DIM], data_t K_weight_dram_A[DIM][DIM], data_t V_weight_dram_A[DIM][DIM], data_t output_dram_A[SEQ_LENGTH][DIM],
        data_t input_dram_B[SEQ_LENGTH][DIM], data_t Q_weight_dram_B[DIM][DIM], data_t K_weight_dram_B[DIM][DIM], data_t V_weight_dram_B[DIM][DIM], data_t output_dram_B[SEQ_LENGTH][DIM])

{
    #pragma HLS interface m_axi port=input_dram_A offset=slave bundle=mem1
    #pragma HLS interface m_axi port=Q_weight_dram_A offset=slave bundle=mem1
    #pragma HLS interface m_axi port=K_weight_dram_A offset=slave bundle=mem1
    #pragma HLS interface m_axi port=V_weight_dram_A offset=slave bundle=mem1
    #pragma HLS interface m_axi port=output_dram_A offset=slave bundle=mem2
    
    #pragma HLS interface m_axi port=input_dram_B offset=slave bundle=mem1
    #pragma HLS interface m_axi port=Q_weight_dram_B offset=slave bundle=mem1
    #pragma HLS interface m_axi port=K_weight_dram_B offset=slave bundle=mem1
    #pragma HLS interface m_axi port=V_weight_dram_B offset=slave bundle=mem1
    #pragma HLS interface m_axi port=output_dram_B offset=slave bundle=mem2

    #pragma HLS allocation function instances= load_embedding limit=1
    #pragma HLS allocation function instances= load_weights limit=1
    #pragma HLS allocation function instances= matmul_large limit=1
    #pragma HLS allocation function instances= matmul_small_1 limit=1
    #pragma HLS allocation function instances= calculate_scale limit=1
    #pragma HLS allocation function instances= softmax limit=1
    #pragma HLS allocation function instances= matmul_small_2 limit=1
    #pragma HLS allocation function instances= store_embedding limit=1

    // //top_A attn in GPT
    // const int groups_A = NUM_HEAD;
    // const int heads_per_group_A = NUM_HEAD / groups_A;
    // load_embedding(input_dram_A, embedding_buffer_1);
    // load_weights(Q_weight_dram_A, Q_weight_buffer);
    // load_weights(K_weight_dram_A, K_weight_buffer);
    // load_weights(V_weight_dram_A, V_weight_buffer);
    // matmul_large(embedding_buffer_1, Q_weight_buffer, Q_buffer);
    // matmul_large(embedding_buffer_1, K_weight_buffer, K_buffer);
    // matmul_large(embedding_buffer_1, V_weight_buffer, V_buffer);



    // for (int g = 0; g < groups_A; g++) {
    //     for (int h = 0; h < heads_per_group_A; h++) {
    //         int head_index = g * heads_per_group_A + h;

    //         matmul_small_1(Q_buffer, K_buffer, scores, head_index);
    //         calculate_scale(scores, scores_2);
    //         softmax(scores_2, scores);
    //         matmul_small_2(scores, V_buffer, embedding_buffer_1, head_index);
    //     }
    // }

    // store_embedding(embedding_buffer_1, output_dram_A);


    // //top_B attn in Llama
    // const int groups_B = 4;
    // const int heads_per_group_B = NUM_HEAD / groups_B;
    // load_embedding(input_dram_B, embedding_buffer_1);
    // load_weights(Q_weight_dram_B, Q_weight_buffer);
    // load_weights(K_weight_dram_B, K_weight_buffer);
    // load_weights(V_weight_dram_B, V_weight_buffer);
    // matmul_large(embedding_buffer_1, Q_weight_buffer, Q_buffer);
    // matmul_large(embedding_buffer_1, K_weight_buffer, K_buffer);
    // matmul_large(embedding_buffer_1, V_weight_buffer, V_buffer);

    // rope(Q_buffer, Q_buffer_rope);
    // rope(K_buffer, K_buffer_rope);
    // for (int g = 0; g < groups_B; g++) {
    //     for (int h = 0; h < heads_per_group_B; h++) {
    //         int head_index = g * heads_per_group_B + h;

    //         matmul_small_1(Q_buffer_rope, K_buffer_rope, scores, head_index);
    //         calculate_scale(scores, scores_2);
    //         softmax(scores_2, scores);
    //         matmul_small_2(scores, V_buffer, embedding_buffer_1, head_index);
    //     }
    // }

    // store_embedding(embedding_buffer_1, output_dram_B);


    // top_A(input_dram_A, Q_weight_dram_A, K_weight_dram_A, V_weight_dram_A, output_dram_A);
    top_B(input_dram_B, Q_weight_dram_B, K_weight_dram_B, V_weight_dram_B, output_dram_B);
}

