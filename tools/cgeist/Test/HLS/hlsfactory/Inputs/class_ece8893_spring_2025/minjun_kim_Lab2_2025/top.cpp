#include "dcl.h"
#include <hls_stream.h>

// Load matrix data into stream
void load_matrix_to_stream(fixed_t matrix[B][N][dk], hls::stream<fixed_t> &stream_out) {
    
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < dk; ++j) {
                #pragma HLS pipeline
                stream_out.write(matrix[b][i][j]);
            }
        }
    }
}

// Store stream data back to matrix
void store_stream_to_matrix(hls::stream<fixed_t> &stream_in, fixed_t matrix[B][N][dv]) {
    
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < dv; ++j) {
                #pragma HLS pipeline
                matrix[b][i][j] = stream_in.read();
            }
        }
    }
}

// Compute Q*K^T product and scale
void compute_qk_product_stream(hls::stream<fixed_t> &q_stream, hls::stream<fixed_t> &k_stream, 
                              hls::stream<fixed_t> &attention_stream, ap_fixed<32, 8> scale) {
    
    for (int b = 0; b < B; ++b) {
        // Buffers to store K
        fixed_t K_buffer[N][dk];
        #pragma HLS array_partition variable=K_buffer dim=2 cyclic factor=4
        
        // Read K into buffer for this batch
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < dk; ++j) {
                #pragma HLS pipeline
                K_buffer[i][j] = k_stream.read();
            }
        }
        
        // Process each row of Q and compute dot products with all rows of K
        for (int i = 0; i < N; ++i) {
            // Buffer for current row of Q
            fixed_t Q_row[dk];
            #pragma HLS array_partition variable=Q_row cyclic factor=4
            
            // Load Q row
            for (int k = 0; k < dk; ++k) {
                #pragma HLS pipeline
                Q_row[k] = q_stream.read();
            }
            
            // Compute dot product for each entry in the attention matrix
            for (int j = 0; j < N; ++j) {
                ap_fixed<32, 8> sum = 0;
                
                #pragma HLS pipeline
                for (int k = 0; k < dk; ++k) {
                    #pragma HLS unroll factor=4
                    sum += Q_row[k] * K_buffer[j][k];
                }
                
                // Scale and write to attention stream
                attention_stream.write(sum * scale);
            }
        }
    }
}

// Apply softmax to attention
void compute_softmax_stream(hls::stream<fixed_t> &input_stream, hls::stream<fixed_t> &output_stream) {
    
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            // Buffer to store row values
            fixed_t row_values[N];
            #pragma HLS array_partition variable=row_values cyclic factor=4
            
            // Read row from stream
            for (int j = 0; j < N; ++j) {
                #pragma HLS pipeline 
                row_values[j] = input_stream.read();
            }
            
            // Find max value in the row
            ap_fixed<32, 8> max_val = row_values[0];
            for (int j = 1; j < N; ++j) {
                #pragma HLS pipeline 
                if (row_values[j] > max_val) {
                    max_val = row_values[j];
                }
            }
            
            // Compute exp(x - max_val) and sum
            ap_fixed<32, 8> sum = 0;
            for (int j = 0; j < N; ++j) {
                #pragma HLS pipeline 
                row_values[j] = hls::exp(row_values[j] - max_val);
                sum += row_values[j];
            }
            
            // Normalize and write to output stream
            for (int j = 0; j < N; ++j) {
                #pragma HLS pipeline
                output_stream.write(row_values[j] / sum);
            }
        }
    }
}

// Compute attention * V product
void compute_attention_value_stream(hls::stream<fixed_t> &attention_stream, hls::stream<fixed_t> &v_stream, 
                                   hls::stream<fixed_t> &output_stream) {
    
    for (int b = 0; b < B; ++b) {
        // Buffer for V matrix
        fixed_t V_buffer[N][dv];
        #pragma HLS array_partition variable=V_buffer dim=2 cyclic factor=4
        
        // Read V into buffer for this batch
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < dv; ++j) {
                #pragma HLS pipeline
                V_buffer[i][j] = v_stream.read();
            }
        }
        
        // Process each row of the attention matrix
        for (int i = 0; i < N; ++i) {
            // Buffer for current row of attention
            fixed_t attention_row[N];
            #pragma HLS array_partition variable=attention_row cyclic factor=4
            
            // Load attention row
            for (int k = 0; k < N; ++k) {
                #pragma HLS pipeline
                attention_row[k] = attention_stream.read();
            }
            
            // Compute output for each column
            for (int j = 0; j < dv; ++j) {
                ap_fixed<32, 8> sum = 0;
                
                #pragma HLS pipeline
                for (int k = 0; k < N; ++k) {
                    #pragma HLS unroll factor=4
                    sum += attention_row[k] * V_buffer[k][j];
                }
                
                // Write result to output stream
                output_stream.write(sum);
            }
        }
    }
}


void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], fixed_t Output[B][N][dv]) {
    // Separate memory interfaces for each input matrix to enable parallel access
    #pragma HLS interface m_axi port=Q offset=slave bundle=mem_q
    #pragma HLS interface m_axi port=K offset=slave bundle=mem_k
    #pragma HLS interface m_axi port=V offset=slave bundle=mem_v
    #pragma HLS interface m_axi port=Output offset=slave bundle=mem_out
    #pragma HLS interface s_axilite port=return
    
    // Declare streams
    hls::stream<fixed_t> q_stream("q_stream");
    hls::stream<fixed_t> k_stream("k_stream");
    hls::stream<fixed_t> v_stream("v_stream");
    hls::stream<fixed_t> attention_stream("attention_stream");
    hls::stream<fixed_t> softmax_output_stream("softmax_output_stream");
    hls::stream<fixed_t> output_stream("output_stream");
    
    // Increase stream depths
    #pragma HLS stream variable=q_stream depth=38400
    #pragma HLS stream variable=k_stream depth=34997
    #pragma HLS stream variable=v_stream depth=38400
    #pragma HLS stream variable=attention_stream depth=2300
    #pragma HLS stream variable=softmax_output_stream depth=2300
    #pragma HLS stream variable=output_stream depth=1
    
    ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);
    
    // Use dataflow to parallelize the entire computation pipeline
    #pragma HLS dataflow
    
    // Load all batches of matrices into streams
    load_matrix_to_stream(Q, q_stream);
    load_matrix_to_stream(K, k_stream);
    load_matrix_to_stream(V, v_stream);

    // Process all batches through the computation pipeline
    compute_qk_product_stream(q_stream, k_stream, attention_stream, scale);
    compute_softmax_stream(attention_stream, softmax_output_stream);
    compute_attention_value_stream(softmax_output_stream, v_stream, output_stream);
    
    // Store all results back to memory
    store_stream_to_matrix(output_stream, Output);
}