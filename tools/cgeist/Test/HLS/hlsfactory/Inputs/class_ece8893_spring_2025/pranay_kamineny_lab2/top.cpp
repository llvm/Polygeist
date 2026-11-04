#include "dcl.h"
#include "hls_stream.h"

// no partitioning arrays in DRAM
// use task-level parallelism, then try loop optimizations 
// simplify or approximate softmax computation to reduce latency
// identify and optimize bottlenecks 

// task 1: compute Q * K^T
void compute_qk_task(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], hls::stream<fixed_t> &qk_out) {
    const ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);
    const int TILE_SIZE = 16;

    #pragma HLS array_partition variable=Q cyclic factor=32 dim=3
    #pragma HLS array_partition variable=K cyclic factor=32 dim=3

    // enable dataflow optimization
    #pragma HLS dataflow

    // local buffer for K
    fixed_t K_local[TILE_SIZE][dk];
    #pragma HLS array_partition variable=K_local cyclic factor=32 dim=2

    for (int b = 0; b < B; ++b) {
        for (int i_tile = 0; i_tile < N; i_tile += TILE_SIZE) {
            for (int j_tile = 0; j_tile < N; j_tile += TILE_SIZE) {
                
                // preload K tiles using burst read
                for (int j_local = 0; j_local < TILE_SIZE && (j_tile + j_local) < N; ++j_local) {
                    #pragma HLS pipeline II=1
                    for (int k = 0; k < dk; k += 32) { 
                        #pragma HLS unroll
                        for (int u = 0; u < 32 && (k + u) < dk; ++u) {
                            K_local[j_local][k + u] = K[b][j_tile + j_local][k + u];
                        }
                    }
                }

                // process tile
                for (int i_local = 0; i_local < TILE_SIZE && (i_tile + i_local) < N; ++i_local) {
                    for (int j_local = 0; j_local < TILE_SIZE && (j_tile + j_local) < N; ++j_local) {
                        #pragma HLS pipeline II=1

                        ap_fixed<32, 8> sum = 0;

                        // dot product calculation with burst access and loop unrolling
                        for (int k = 0; k < dk; k += 32) {
                            #pragma HLS unroll
                            for (int u = 0; u < 32 && (k + u) < dk; ++u) {
                                sum += Q[b][i_tile + i_local][k + u] * K_local[j_local][k + u];
                            }
                        }

                        // scale and write to output stream
                        qk_out.write(sum * scale);
                    }
                }
            }
        }
    }
}

// task 2: apply softmax 
void softmax_task(hls::stream<fixed_t> &qk_in, hls::stream<fixed_t> &softmax_out) {
    #pragma HLS DATAFLOW
    #pragma HLS INLINE

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            #pragma HLS PIPELINE II=1

            // buffer to store row for max calculation
            fixed_t row_buffer[N];
            #pragma HLS ARRAY_PARTITION variable=row_buffer complete dim=1

            // read row and find max
            ap_fixed<32, 8> max_val = -INFINITY;
            for (int j = 0; j < N; ++j) {
                #pragma HLS UNROLL
                row_buffer[j] = qk_in.read();
                if (row_buffer[j] > max_val) {
                    max_val = row_buffer[j];
                }
            }

            // compute exp and sum
            ap_fixed<32, 8> sum = 0;
            fixed_t exp_buffer[N];
            #pragma HLS ARRAY_PARTITION variable=exp_buffer complete dim=1

            for (int j = 0; j < N; ++j) {
                #pragma HLS UNROLL
                exp_buffer[j] = hls::exp(row_buffer[j] - max_val);
                sum += exp_buffer[j];
            }

            // normalize and write to stream
            for (int j = 0; j < N; ++j) {
                #pragma HLS UNROLL
                softmax_out.write(exp_buffer[j] / sum);
            }
        }
    }
}

// task 3: compute attention * V
void compute_attention_task(hls::stream<fixed_t> &softmax_in, fixed_t V[B][N][dv], fixed_t Output[B][N][dv]) {
    #pragma HLS array_partition variable=V cyclic factor=8 dim=3

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            fixed_t attention_row[N];
            
            // read and store softmax row
            for (int j = 0; j < N; ++j) {
                #pragma HLS pipeline II=1
                attention_row[j] = softmax_in.read();
            }

            for (int j = 0; j < dv; ++j) {
                ap_fixed<32, 8> sum = 0;

                // compute weighted sum with V
                #pragma HLS pipeline II=1
                #pragma HLS unroll factor=8
                for (int k = 0; k < N; ++k) {
                    sum += attention_row[k] * V[b][k][j];
                }
                Output[b][i][j] = sum;
            }
        }
    }
}


// want to enforce AXI burst mode and utilize full DRAM bandwidth
void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], fixed_t Output[B][N][dv]) {
#pragma HLS interface m_axi port=Q offset=slave bundle=mem1
#pragma HLS interface m_axi port=K offset=slave bundle=mem1
#pragma HLS interface m_axi port=V offset=slave bundle=mem1
#pragma HLS interface m_axi port=Output offset=slave bundle=mem2
#pragma HLS interface s_axilite port=return

    fixed_t Q_buf[B][N][dk];
    fixed_t K_buf[B][N][dk];
    fixed_t V_buf[B][N][dv];
#pragma HLS array_partition variable=Q_buf cyclic factor=4 dim=3
#pragma HLS array_partition variable=K_buf cyclic factor=4 dim=3
#pragma HLS array_partition variable=V_buf cyclic factor=4 dim=3


    // copy Q, K, and V into local buffers in BRAM
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < dk; ++k) {
#pragma HLS unroll factor=4
                Q_buf[b][i][k] = Q[b][i][k]; 
                K_buf[b][i][k] = K[b][i][k]; 
            }
            for (int d = 0; d < dv; ++d) {
#pragma HLS unroll factor=4
                V_buf[b][i][d] = V[b][i][d]; 
            }
        }
    }

    // task-level parallelism with streams
    hls::stream<fixed_t> qk_stream;
    hls::stream<fixed_t> softmax_stream;
#pragma HLS stream variable=qk_stream depth=16
#pragma HLS stream variable=softmax_stream depth=16

#pragma HLS dataflow
    // launch 3 tasks with dataflow 
    compute_qk_task(Q_buf, K_buf, qk_stream);
    softmax_task(qk_stream, softmax_stream);
    compute_attention_task(softmax_stream, V_buf, Output);
}
