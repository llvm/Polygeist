// FPGA_ECE8893 Lab 2: Efficient Implementation of Attention Computation
// Name: Gaurav Singh Chandrabhan
// Note: Only the top.cpp file is being changed

#include "dcl.h"
#include <hls_stream.h>

typedef hls::stream<fixed_t> stream_t;

// softmax_HLS()
//   Reads B×N×N values from attention_in and computes the softmax
//   row-by-row. The results are written in order to attention_out.
//
void softmax_HLS(stream_t &attention_in, stream_t &attention_out) {
    #pragma HLS INLINE off

    fixed_t attention[B][N][N];

    // Read attention scores from the stream into a local buffer.
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                #pragma HLS PIPELINE II=1
                attention[b][i][j] = attention_in.read();
            }
        }
    }

    // For each row, compute the softmax.
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            ap_fixed<32, 8> max_val = attention[b][i][0];
            // Find maximum value for numerical stability.
            for (int j = 1; j < N; ++j) {
                #pragma HLS PIPELINE II=1
                if (attention[b][i][j] > max_val) {
                    max_val = attention[b][i][j];
                }
            }
            ap_fixed<32, 8> sum = 0;
            // Exponentiate and accumulate sum.
            for (int j = 0; j < N; ++j) {
                #pragma HLS PIPELINE II=1
                attention[b][i][j] = hls::exp(attention[b][i][j] - max_val);
                sum += attention[b][i][j];
            }
            // Normalize and write results to the stream.
            for (int j = 0; j < N; ++j) {
                #pragma HLS PIPELINE II=1
                attention[b][i][j] /= sum;
                attention_out.write(attention[b][i][j]);
            }
        }
    }
}

// compute_attention_HLS()
//   Compute the Attention mechanism in three steps:
//     1. Compute Q * K^T (scaled) → attention matrix.
//     2. Apply softmax on the attention matrix.
//     3. Compute the final output = softmax(attention) * V.
//
void compute_attention_HLS(fixed_t Q[B][N][dk],
                           fixed_t K[B][N][dk],
                           fixed_t V[B][N][dv],
                           fixed_t Output[B][N][dv]) {
#pragma HLS INTERFACE m_axi port=Q offset=slave bundle=mem1 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=K offset=slave bundle=mem1 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=V offset=slave bundle=mem1 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=Output offset=slave bundle=mem2 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS DATAFLOW

    // Declare streams with increased depths to avoid deadlock
    stream_t attention_in_stream, attention_out_stream, output_stream;
    #pragma HLS STREAM variable=attention_in_stream depth=512
    #pragma HLS STREAM variable=attention_out_stream depth=512
    #pragma HLS STREAM variable=output_stream depth=256

    // Local buffers for Q, K, V and for storing the softmax result.
    fixed_t Q_buf[B][N][dk];
    fixed_t K_buf[B][N][dk];
    fixed_t V_buf[B][N][dv];
    fixed_t softmax_buf[B][N][N];  // To hold the full attention matrix after softmax

    #pragma HLS ARRAY_PARTITION variable=Q_buf dim=3 complete
    #pragma HLS ARRAY_PARTITION variable=K_buf dim=3 complete
    #pragma HLS ARRAY_PARTITION variable=V_buf dim=3 complete

    // Bind buffers to BRAM to avoid assumptions
    #pragma HLS BIND_STORAGE variable=Q_buf type=RAM_2P impl=BRAM
    #pragma HLS BIND_STORAGE variable=K_buf type=RAM_2P impl=BRAM
    #pragma HLS BIND_STORAGE variable=V_buf type=RAM_2P impl=BRAM
    #pragma HLS BIND_STORAGE variable=softmax_buf type=RAM_2P impl=BRAM

    // Load Q, K, V from DRAM into local buffers.
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < dk; ++k) {
                #pragma HLS PIPELINE II=1
                Q_buf[b][i][k] = Q[b][i][k];
                K_buf[b][i][k] = K[b][i][k];
            }
            for (int k = 0; k < dv; ++k) {
                #pragma HLS PIPELINE II=1
                V_buf[b][i][k] = V[b][i][k];
            }
        }
    }
    ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);
    // 1. Compute Q * K^T (with scaling) and stream the result.
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                ap_fixed<32, 8> sum = 0;
                for (int k = 0; k < dk; ++k) {
                    #pragma HLS PIPELINE II=1
                    sum += Q_buf[b][i][k] * K_buf[b][j][k];
                }
                // Multiply by scale factor
                sum = sum * scale;
                attention_in_stream.write(sum);
            }
        }
    }

    // 2. Apply softmax
    softmax_HLS(attention_in_stream, attention_out_stream);

    // Capture softmax results into a local buffer.
    // The softmax output is produced in the same order as the attention matrix BxNxN
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                #pragma HLS PIPELINE II=1
                softmax_buf[b][i][j] = attention_out_stream.read();
            }
        }
    }

    // 3. Compute Attention * V.
    // For each output element, perform a dot product between a softmax row and the corresponding column of V.
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < dv; ++j) {
                ap_fixed<32, 8> sum = 0;
                for (int k = 0; k < N; ++k) {
                    #pragma HLS PIPELINE II=1
                    sum += softmax_buf[b][i][k] * V_buf[b][k][j];
                }
                // Stream the output for BRAM to DRAM
                output_stream.write(sum);
            }
        }
    }

    // Write the computed output to DRAM
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < dv; ++j) {
                #pragma HLS PIPELINE II=1
                Output[b][i][j] = output_stream.read();
            }
        }
    }
}
