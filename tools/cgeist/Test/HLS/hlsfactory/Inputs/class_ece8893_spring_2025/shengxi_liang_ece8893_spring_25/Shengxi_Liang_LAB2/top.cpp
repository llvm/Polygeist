#include "dcl.h"
#include <hls_stream.h>

// Define row-based types for streaming data between processing elements
typedef struct {
    fixed_t data[N];
} qk_row_t;

typedef struct {
    fixed_t data[dv];
} v_row_t;

typedef struct {
    fixed_t data[N];
} attention_row_t;

typedef struct {
    fixed_t data[dv];
} output_row_t;

// Stage 1: QK^T Computation
void compute_qk_task(
    const fixed_t Q[B][N][dk], 
    const fixed_t K[B][N][dk],
    hls::stream<attention_row_t> &qk_stream) {
    
    // Local buffer for K to ensure proper burst read
    fixed_t K_local[N][dk];
    #pragma HLS ARRAY_PARTITION variable=V_local dim=2 factor=2 cyclic
    
    // Scaling factor
    const ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);
    
    // Process each batch
    for (int b = 0; b < B; ++b) {
        // Load K for current batch to local memory for reuse
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < dk; ++k) {
                #pragma HLS PIPELINE II=1
                K_local[j][k] = K[b][j][k];
            }
        }
        
        // Compute Q*K^T row by row (N rows)
        for (int i = 0; i < N; ++i) {
            attention_row_t qk_row;
            
            // Compute one row of Q*K^T
            for (int j = 0; j < N; ++j) {
                #pragma HLS PIPELINE II=1
                ap_fixed<32, 8> sum = 0;
                for (int k = 0; k < dk; ++k) {
                    #pragma HLS UNROLL factor=16
                    sum += Q[b][i][k] * K_local[j][k];
                }
                qk_row.data[j] = sum * scale;
            }
            
            // Stream out the computed row
            qk_stream.write(qk_row);
        }
    }
}


// Stage 2: Softmax Computation with numerical stability
void softmax_task(
    hls::stream<attention_row_t> &qk_stream,
    hls::stream<attention_row_t> &softmax_stream) {
    
    // Process each batch
    for (int b = 0; b < B; ++b) {
        // Process each row
        for (int i = 0; i < N; ++i) {
            // Read input row from QK computation
            attention_row_t row_in = qk_stream.read();
            attention_row_t row_out;
            
            // Find maximum value in row (for numerical stability)
            fixed_t max_val = row_in.data[0];
            for (int j = 1; j < N; ++j) {
                #pragma HLS PIPELINE II=1
                if (row_in.data[j] > max_val) {
                    max_val = row_in.data[j];
                }
            }
            
            // Compute exp(x - max_val) for each element
            fixed_t sum = 0;
            fixed_t exp_vals[N];
            #pragma HLS ARRAY_PARTITION variable=exp_vals cyclic factor=2
            
            for (int j = 0; j < N; ++j) {
                #pragma HLS PIPELINE II=1
                exp_vals[j] = hls::exp(row_in.data[j] - max_val);
                sum += exp_vals[j];
            }
            
            // Normalize by sum
            for (int j = 0; j < N; ++j) {
                #pragma HLS PIPELINE II=1
                row_out.data[j] = exp_vals[j] / sum;
            }
            
            // Stream out the softmax result
            softmax_stream.write(row_out);
        }
    }
}

void attention_v_task(
    const fixed_t V[B][N][dv],
    hls::stream<attention_row_t> &softmax_stream,
    fixed_t Output[B][N][dv]) {
        fixed_t V_local[N][dv];
 
            

    // Process each batch
    for (int b = 0; b < B; ++b) {
        // Load V for current batch
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < dv; ++k) {
                #pragma HLS PIPELINE II=1
                V_local[j][k] = V[b][j][k];
            }
        }
        
        // Compute output rows
        for (int i = 0; i < N; ++i) {
            // Read softmax row
            attention_row_t attn_weights = softmax_stream.read();
            
            // Compute output[i] = attention[i] * V
            for (int j = 0; j < dv; ++j) {
                #pragma HLS PIPELINE II=1
                ap_fixed<32, 8> sum = 0;
                for (int k = 0; k < N; ++k) {
                    #pragma HLS UNROLL factor=2
                    sum += attn_weights.data[k] * V_local[k][j];
                }
                Output[b][i][j] = sum;
            }
        }
    }
}
// Main attention function with dataflow - using exact signature from host.cpp
void compute_attention_HLS(
    fixed_t Q[B][N][dk], 
    fixed_t K[B][N][dk], 
    fixed_t V[B][N][dv], 
    fixed_t Output[B][N][dv]) {
    
    #pragma HLS interface m_axi port=Q offset=slave bundle=gmem1 
    #pragma HLS interface m_axi port=K offset=slave bundle=gmem2 
    #pragma HLS interface m_axi port=V offset=slave bundle=gmem3 
    #pragma HLS interface m_axi port=Output offset=slave bundle=gmem4 
    #pragma HLS interface s_axilite port=return
    
    // Define streams to connect processing elements with adequate depth
    hls::stream<attention_row_t> qk_stream;
    hls::stream<attention_row_t> softmax_stream;
    
    #pragma HLS STREAM variable=qk_stream depth=4
    #pragma HLS STREAM variable=softmax_stream depth=4
    
    // Enable dataflow for task-level parallelism
    #pragma HLS dataflow
    
    // Task 1: Compute Q*K^T
    compute_qk_task(Q, K, qk_stream);
    
    // Task 2: Apply softmax operation
    softmax_task(qk_stream, softmax_stream);
    
    // Task 3: Compute Attention*V
    attention_v_task(V, softmax_stream, Output);
}