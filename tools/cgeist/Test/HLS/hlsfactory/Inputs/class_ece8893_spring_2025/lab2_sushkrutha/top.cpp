#include "dcl.h"
#include <hls_stream.h>
#include <cmath>

// Approximate softmax computation for a 3D matrix
void softmax_HLS_approx(fixed_t matrix[B][N][N]) {
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            #pragma HLS PIPELINE II=1
            ap_fixed<32, 8> max_val = matrix[b][i][0]; // Initialize max value with the first element
            for (int j = 1; j < N; ++j) {
                // Find the maximum value in the current row
                if (matrix[b][i][j] > max_val) {
                    max_val = matrix[b][i][j];
                }
            }

            ap_fixed<32, 8> sum = 0; // Initialize sum for normalization
            for (int j = 0; j < N; ++j) {
                #pragma HLS UNROLL
                // Compute exponential of (value - max) for numerical stability
                matrix[b][i][j] = hls::exp(matrix[b][i][j] - max_val);
                sum += matrix[b][i][j]; // Accumulate sum for normalization
            }

            for (int j = 0; j < N; ++j) {
                #pragma HLS UNROLL
                // Normalize by dividing by the sum
                matrix[b][i][j] /= sum;
            }
        }
    }
}

// Compute dot product of Q and K^T, followed by softmax
void compute_dot_product(hls::stream<fixed_t>& Q_stream, hls::stream<fixed_t>& K_stream, hls::stream<fixed_t>& attention_stream, ap_fixed<32, 8> scale) {
    fixed_t Q_local[B][N][dk]; // Local storage for Q matrix
    fixed_t K_local[B][N][dk]; // Local storage for K matrix
    fixed_t attention_local[B][N][N]; // Local storage for attention scores

    // Load Q and K matrices from streams into local memory
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < dk; ++k) {
                #pragma HLS PIPELINE II=1
                Q_local[b][i][k] = Q_stream.read();
                K_local[b][i][k] = K_stream.read();
            }
        }
    }

    // Compute dot product: Q * K^T
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                #pragma HLS PIPELINE II=1
                ap_fixed<32, 8> sum = 0; // Initialize sum for dot product
                for (int k = 0; k < dk; ++k) {
                    // Accumulate dot product
                    sum += Q_local[b][i][k] * K_local[b][j][k];
                }
                // Scale the result
                attention_local[b][i][j] = sum * scale;
            }
        }
    }

    // Apply approximate softmax to attention scores
    softmax_HLS_approx(attention_local);

    // Write attention scores to output stream
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                #pragma HLS PIPELINE II=1
                attention_stream.write(attention_local[b][i][j]);
            }
        }
    }
}

// Compute weighted sum of attention scores and V
void compute_weighted_sum(hls::stream<fixed_t>& attention_stream, hls::stream<fixed_t>& V_stream, hls::stream<fixed_t>& Output_stream) {
    fixed_t attention_local[B][N][N]; // Local storage for attention scores
    fixed_t V_local[B][N][dv]; // Local storage for V matrix
    fixed_t Output_local[B][N][dv]; // Local storage for output matrix

    // Load attention scores and V matrix from streams into local memory
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

    // Compute weighted sum: attention * V
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < dv; ++j) {
                #pragma HLS PIPELINE II=1
                ap_fixed<32, 8> sum = 0; // Initialize sum for weighted sum
                for (int k = 0; k < N; ++k) {
                    // Accumulate weighted sum
                    sum += attention_local[b][i][k] * V_local[b][k][j];
                }
                Output_local[b][i][j] = sum; // Store result
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
    #pragma HLS INTERFACE m_axi port=Q offset=slave bundle=mem1 depth=1024 latency=16 max_read_burst_length=32 num_read_outstanding=32
    #pragma HLS INTERFACE m_axi port=K offset=slave bundle=mem1 depth=1024 latency=16 max_read_burst_length=32 num_read_outstanding=32
    #pragma HLS INTERFACE m_axi port=V offset=slave bundle=mem2 depth=1024 latency=16 max_read_burst_length=32 num_read_outstanding=32
    #pragma HLS INTERFACE m_axi port=Output offset=slave bundle=mem2 depth=1024 latency=16 max_write_burst_length=32 num_write_outstanding=32
    #pragma HLS INTERFACE s_axilite port=return

    #pragma HLS DATAFLOW

    hls::stream<fixed_t> Q_stream, K_stream, V_stream, attention_stream, Output_stream;

    // Load Q, K, and V matrices into streams using AXI burst mode
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            #pragma HLS PIPELINE II=1
            for (int k = 0; k < dk; ++k) {
                Q_stream.write(Q[b][i][k]);
                K_stream.write(K[b][i][k]);
            }
        }
    }
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            #pragma HLS PIPELINE II=1
            for (int j = 0; j < dv; ++j) {
                V_stream.write(V[b][i][j]);
            }
        }
    }

    // Compute dot product and softmax
    ap_fixed<32, 8> scale = 1.0 / std::sqrt((float)dk); // Scaling factor
    compute_dot_product(Q_stream, K_stream, attention_stream, scale);

    // Compute weighted sum of attention scores and V
    compute_weighted_sum(attention_stream, V_stream, Output_stream);

    // Write output results from stream to memory using AXI burst mode
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            #pragma HLS PIPELINE II=1
            for (int j = 0; j < dv; ++j) {
                Output[b][i][j] = Output_stream.read();
            }
        }
    }
}
