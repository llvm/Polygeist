// Name: Shree Sahitya Balaje

#include "dcl.h"
#include <hls_stream.h>
#include <cmath>

// Function to compute softmax with approximation
void softmax_HLS(fixed_t matrix[B][N][N]) {
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            #pragma HLS PIPELINE II=1
            ap_fixed<32, 8> max_val = matrix[b][i][0];
            for (int j = 1; j < N; ++j) {
                #pragma HLS UNROLL
                if (matrix[b][i][j] > max_val) {
                    max_val = matrix[b][i][j];
                }
            }

            ap_fixed<32, 8> sum = 0;
            for (int j = 0; j < N; ++j) {
                #pragma HLS UNROLL
                matrix[b][i][j] = hls::exp(matrix[b][i][j] - max_val);
                sum += matrix[b][i][j];
            }

            for (int j = 0; j < N; ++j) {
                #pragma HLS UNROLL
                matrix[b][i][j] /= sum;
            }
        }
    }
}

// Function to compute Q * K^T and softmax
void compute_QK_T_softmax(hls::stream<fixed_t>& Q_stream, hls::stream<fixed_t>& K_stream, hls::stream<fixed_t>& attention_stream, ap_fixed<32, 8> scale) {
    fixed_t Q_local[B][N][dk];
    fixed_t K_local[B][N][dk];
    fixed_t attention_local[B][N][N];

    // Load Q and K from streams into local memory
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < dk; ++k) {
                #pragma HLS PIPELINE II=1
                Q_local[b][i][k] = Q_stream.read();
                K_local[b][i][k] = K_stream.read();
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
                    #pragma HLS UNROLL
                    sum += Q_local[b][i][k] * K_local[b][j][k];
                }
                attention_local[b][i][j] = sum * scale;
            }
        }
    }

    // Apply softmax
    softmax_HLS(attention_local);

    // Write attention results to stream
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                #pragma HLS PIPELINE II=1
                attention_stream.write(attention_local[b][i][j]);
            }
        }
    }
}

// Function to compute Attention * V
void compute_attention_V(hls::stream<fixed_t>& attention_stream, hls::stream<fixed_t>& V_stream, hls::stream<fixed_t>& Output_stream) {
    fixed_t attention_local[B][N][N];
    fixed_t V_local[B][N][dv];
    fixed_t Output_local[B][N][dv];

    // Load attention and V from streams into local memory
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                #pragma HLS PIPELINE II=1
                attention_local[b][i][j] = attention_stream.read();
            }
        }
    }
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < dv; ++j) {
                #pragma HLS PIPELINE II=1
                V_local[b][i][j] = V_stream.read();
            }
        }
    }

    // Compute Attention * V
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < dv; ++j) {
                #pragma HLS PIPELINE II=1
                ap_fixed<32, 8> sum = 0;
                for (int k = 0; k < N; ++k) {
                    #pragma HLS UNROLL
                    sum += attention_local[b][i][k] * V_local[b][k][j];
                }
                Output_local[b][i][j] = sum;
            }
        }
    }

    // Write output results to stream
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < dv; ++j) {
                #pragma HLS PIPELINE II=1
                Output_stream.write(Output_local[b][i][j]);
            }
        }
    }
}

// Top-level function for attention computation
void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], fixed_t Output[B][N][dv]) {
    #pragma HLS INTERFACE m_axi port=Q offset=slave bundle=mem1 depth=1024
    #pragma HLS INTERFACE m_axi port=K offset=slave bundle=mem1 depth=1024
    #pragma HLS INTERFACE m_axi port=V offset=slave bundle=mem2 depth=1024
    #pragma HLS INTERFACE m_axi port=Output offset=slave bundle=mem2 depth=1024
    #pragma HLS INTERFACE s_axilite port=return

    #pragma HLS DATAFLOW

    hls::stream<fixed_t> Q_stream, K_stream, V_stream, attention_stream, Output_stream;

    // Load Q, K, and V into streams
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < dk; ++k) {
                #pragma HLS PIPELINE II=1
                Q_stream.write(Q[b][i][k]);
                K_stream.write(K[b][i][k]);
            }
        }
    }
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < dv; ++j) {
                #pragma HLS PIPELINE II=1
                V_stream.write(V[b][i][j]);
            }
        }
    }

    // Compute Q * K^T and softmax
    ap_fixed<32, 8> scale = 1.0 / std::sqrt((float)dk);
    compute_QK_T_softmax(Q_stream, K_stream, attention_stream, scale);

    // Compute Attention * V
    compute_attention_V(attention_stream, V_stream, Output_stream);

    // Write output results from stream to memory
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < dv; ++j) {
                #pragma HLS PIPELINE II=1
                Output[b][i][j] = Output_stream.read();
            }
        }
    }
}