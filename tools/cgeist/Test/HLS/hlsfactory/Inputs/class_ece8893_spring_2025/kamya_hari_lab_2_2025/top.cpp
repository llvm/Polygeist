#include "dcl.h"
#include <hls_math.h>

void softmax_HLS(fixed_t matrix[B][N][N]) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            fixed_t max_val = matrix[b][i][0];
            
            // Find max value
            for (int j = 1; j < N; ++j) {
                #pragma HLS PIPELINE II=1
                if (matrix[b][i][j] > max_val) {
                    max_val = matrix[b][i][j];
                }
            }
            
            ap_fixed<32, 8> sum = 0;
            
            // Compute exponent approximation and sum
            for (int j = 0; j < N; ++j) {
                #pragma HLS PIPELINE II=1
                ap_fixed<32, 8> x = matrix[b][i][j] - max_val;
                matrix[b][i][j] = hls::exp(x);
                sum += matrix[b][i][j];
            }
            
            // Normalize
            ap_fixed<32, 8> inv_sum = 1 / sum;
            for (int j = 0; j < N; ++j) {
                #pragma HLS PIPELINE II=1
                matrix[b][i][j] *= inv_sum;
            }
        }
    }
}

void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], fixed_t Output[B][N][dv]) {
#pragma HLS INTERFACE m_axi port=Q offset=slave bundle=mem1 depth=1024 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=K offset=slave bundle=mem1 depth=1024 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=V offset=slave bundle=mem1 depth=1024 num_read_outstanding=32
#pragma HLS INTERFACE m_axi port=Output offset=slave bundle=mem2 depth=1024 num_write_outstanding=32
#pragma HLS INTERFACE s_axilite port=return

    fixed_t Q_local[B][N][dk], K_local[B][N][dk], V_local[B][N][dv], attention_local[B][N][N];
    #pragma HLS ARRAY_PARTITION variable=attention_local cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=Q_local cyclic factor=8 dim=3
    #pragma HLS ARRAY_PARTITION variable=K_local cyclic factor=8 dim=3
    #pragma HLS ARRAY_PARTITION variable=V_local cyclic factor=8 dim=3
    
    ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);
    
    // Load data into local buffers
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < dk; ++k) {
                #pragma HLS PIPELINE II=1
                Q_local[b][i][k] = Q[b][i][k];
                K_local[b][i][k] = K[b][i][k];
            }
            for (int j = 0; j < dv; ++j) {
                #pragma HLS PIPELINE II=1
                V_local[b][i][j] = V[b][i][j];
            }
        }
    }
    
    // Compute Q * K^T
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                #pragma HLS PIPELINE II=1
                ap_fixed<32, 8> sum = 0;
                for (int k = 0; k < dk; ++k) {
                    #pragma HLS UNROLL factor=8
                    sum += Q_local[b][i][k] * K_local[b][j][k];
                }
                attention_local[b][i][j] = sum * scale;
            }
        }
        
        // Apply softmax
        softmax_HLS(attention_local);
    }
    
    // Compute Attention * V
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < dv; ++j) {
                #pragma HLS PIPELINE II=1
                ap_fixed<32, 8> sum = 0;
                for (int k = 0; k < N; ++k) {
                    #pragma HLS UNROLL factor=8
                    sum += attention_local[b][i][k] * V_local[b][k][j];
                }
                Output[b][i][j] = sum;
            }
        }
    }
}

