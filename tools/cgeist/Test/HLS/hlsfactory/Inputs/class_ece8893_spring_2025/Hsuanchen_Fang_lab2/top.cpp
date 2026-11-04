#include "dcl.h"
#include "hls_stream.h"
#include "hls_math.h"

// Function to load Q and K matrices into streams
void load_qk(fixed_t Q[B][N][dk], fixed_t K[B][N][dk],
             hls::stream<fixed_t> &q_stream, hls::stream<fixed_t> &k_stream, int batch) {
#pragma HLS INLINE off

    QK_LOAD: for (int i = 0; i < N; ++i) {
        for (int j = 0; j < dk; ++j) {
#pragma HLS PIPELINE II=1
            q_stream.write(Q[batch][i][j]);
            k_stream.write(K[batch][i][j]);
        }
    }
}

// Split and buffer Q and K for later processing
void split_qk(hls::stream<fixed_t> &q_stream, hls::stream<fixed_t> &k_stream,
              hls::stream<fixed_t> &q_first_stream, hls::stream<fixed_t> &k_first_stream,
              hls::stream<fixed_t> &q_second_stream, hls::stream<fixed_t> &k_second_stream) {
#pragma HLS INLINE off

    // Read all Q and K values and split them
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < dk; ++j) {
#pragma HLS PIPELINE II=1
            fixed_t q_val = q_stream.read();
            fixed_t k_val = k_stream.read();
            
            if (j < dk/2) {
                q_first_stream.write(q_val);
                k_first_stream.write(k_val);
            } else {
                q_second_stream.write(q_val);
                k_second_stream.write(k_val);
            }
        }
    }
}

// Compute first half of QK^T dot products
void compute_qk_part1(hls::stream<fixed_t> &q_first_stream, hls::stream<fixed_t> &k_first_stream,
                    hls::stream<fixed_t> &qk_part1_stream) {
#pragma HLS INLINE off

    // Use local buffers for first half of Q and K
    fixed_t Q_buf[N][dk/2];
#pragma HLS RESOURCE variable=Q_buf core=RAM_2P_BRAM
#pragma HLS ARRAY_PARTITION variable=Q_buf cyclic factor=14 dim=2

    fixed_t K_buf[N][dk/2];
#pragma HLS RESOURCE variable=K_buf core=RAM_2P_BRAM
#pragma HLS ARRAY_PARTITION variable=K_buf cyclic factor=14 dim=2

    // Load first half of Q and K
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < dk/2; ++j) {
#pragma HLS PIPELINE II=1
            Q_buf[i][j] = q_first_stream.read();
            K_buf[i][j] = k_first_stream.read();
        }
    }

    // Compute partial dot products for first half
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
#pragma HLS PIPELINE II=1
            ap_fixed<32, 8> dot_product = 0;
            
            for (int k = 0; k < dk/2; k += 14) {
#pragma HLS UNROLL
                ap_fixed<32, 8> partial_sum = 0;
                for (int kk = 0; kk < 14 && (k + kk) < dk/2; ++kk) {
                    partial_sum += Q_buf[i][k + kk] * K_buf[j][k + kk];
                }
                dot_product += partial_sum;
            }
            
            qk_part1_stream.write(dot_product);
        }
    }
}

// Compute second half of QK^T and merge results
void compute_qk_part2(hls::stream<fixed_t> &q_second_stream, hls::stream<fixed_t> &k_second_stream,
                    hls::stream<fixed_t> &qk_part1_stream, hls::stream<fixed_t> &attention_matrix_stream,
                    fixed_t scale) {
#pragma HLS INLINE off

    // Use local buffers for second half of Q and K
    fixed_t Q_buf[N][dk/2];
#pragma HLS RESOURCE variable=Q_buf core=RAM_2P_BRAM
#pragma HLS ARRAY_PARTITION variable=Q_buf cyclic factor=14 dim=2

    fixed_t K_buf[N][dk/2];
#pragma HLS RESOURCE variable=K_buf core=RAM_2P_BRAM
#pragma HLS ARRAY_PARTITION variable=K_buf cyclic factor=14 dim=2

    // Load second half of Q and K
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < dk/2; ++j) {
#pragma HLS PIPELINE II=1
            Q_buf[i][j] = q_second_stream.read();
            K_buf[i][j] = k_second_stream.read();
        }
    }

    // Compute partial dot products for second half and combine with first half
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
#pragma HLS PIPELINE II=1
            ap_fixed<32, 8> dot_product = 0;
            
            for (int k = 0; k < dk/2; k += 14) {
#pragma HLS UNROLL
                ap_fixed<32, 8> partial_sum = 0;
                for (int kk = 0; kk < 14 && (k + kk) < dk/2; ++kk) {
                    partial_sum += Q_buf[i][k + kk] * K_buf[j][k + kk];
                }
                dot_product += partial_sum;
            }
            
            // Read partial result from first half and combine
            ap_fixed<32, 8> part1 = qk_part1_stream.read();
            attention_matrix_stream.write((part1 + dot_product) * scale);
        }
    }
}

// Apply softmax
void softmax_stream(hls::stream<fixed_t> &in_stream, hls::stream<fixed_t> &out_stream) {
#pragma HLS INLINE off

    for (int i = 0; i < N; ++i) {
        // Buffer to store row values
        fixed_t row[N];
#pragma HLS RESOURCE variable=row core=RAM_2P_BRAM
#pragma HLS ARRAY_PARTITION variable=row cyclic factor=14 dim=1
        
        // Read the entire row
        for (int j = 0; j < N; ++j) {
#pragma HLS PIPELINE II=1
            row[j] = in_stream.read();
        }
        
        // Find max value exactly as in original code
        ap_fixed<32, 8> max_val = row[0];
        for (int j = 1; j < N; ++j) {
#pragma HLS PIPELINE II=1
            if (row[j] > max_val) {
                max_val = row[j];
            }
        }
        
        // Apply exp and sum
        ap_fixed<32, 8> sum = 0;
        for (int j = 0; j < N; ++j) {
#pragma HLS PIPELINE II=1
            row[j] = hls::exp(row[j] - max_val);
            sum += row[j];
        }
        
        // Normalize
        for (int j = 0; j < N; ++j) {
#pragma HLS PIPELINE II=1
            out_stream.write(row[j] / sum);
        }
    }
}

// Process V in tiles to balance performance and BRAM usage
void compute_output_tiles(hls::stream<fixed_t> &attention_stream, 
                         fixed_t V[B][N][dv], fixed_t Output[B][N][dv], int batch) {
#pragma HLS INLINE off

    // Buffer for attention values
    fixed_t attention_values[N][N];
#pragma HLS RESOURCE variable=attention_values core=RAM_2P_BRAM
#pragma HLS ARRAY_PARTITION variable=attention_values cyclic factor=14 dim=2
    
    // Load attention values once
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
#pragma HLS PIPELINE II=1
            attention_values[i][j] = attention_stream.read();
        }
    }
    
    // Process in tiles to reduce BRAM usage
    const int TILE_SIZE = 32;
    
    for (int j_start = 0; j_start < dv; j_start += TILE_SIZE) {
        // Determine actual tile size (handling boundary case)
        int j_end = (j_start + TILE_SIZE > dv) ? dv : j_start + TILE_SIZE;
        int tile_width = j_end - j_start;
        
        // Local buffer for V tile
        fixed_t V_tile[N][TILE_SIZE];
#pragma HLS RESOURCE variable=V_tile core=RAM_2P_BRAM
#pragma HLS ARRAY_PARTITION variable=V_tile cyclic factor=14 dim=2
        
        // Load V tile
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < tile_width; ++j) {
#pragma HLS PIPELINE II=1
                V_tile[i][j] = V[batch][i][j_start + j];
            }
        }
        
        // Compute output for this tile
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < tile_width; ++j) {
#pragma HLS PIPELINE II=1
                ap_fixed<32, 8> sum = 0;
                
                for (int k = 0; k < N; k += 14) {
#pragma HLS UNROLL
                    ap_fixed<32, 8> partial_sum = 0;
                    for (int kk = 0; kk < 14 && (k + kk) < N; ++kk) {
                        partial_sum += attention_values[i][k + kk] * V_tile[k + kk][j];
                    }
                    sum += partial_sum;
                }
                
                Output[batch][i][j_start + j] = sum;
            }
        }
    }
}

void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], 
                         fixed_t V[B][N][dv], fixed_t Output[B][N][dv]) {
#pragma HLS interface m_axi port=Q offset=slave bundle=gmem0 max_read_burst_length=64 max_write_burst_length=64
#pragma HLS interface m_axi port=K offset=slave bundle=gmem1 max_read_burst_length=64 max_write_burst_length=64
#pragma HLS interface m_axi port=V offset=slave bundle=gmem2 max_read_burst_length=64 max_write_burst_length=64
#pragma HLS interface m_axi port=Output offset=slave bundle=gmem3 max_read_burst_length=64 max_write_burst_length=64
#pragma HLS interface s_axilite port=return

    const ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);

    // Process each batch
    BATCH: for (int b = 0; b < B; ++b) {
#pragma HLS DATAFLOW

        // Create streams for data transfer between stages
        hls::stream<fixed_t> q_stream("q_stream");
        hls::stream<fixed_t> k_stream("k_stream");
        hls::stream<fixed_t> q_first_stream("q_first_stream");
        hls::stream<fixed_t> k_first_stream("k_first_stream");
        hls::stream<fixed_t> q_second_stream("q_second_stream");
        hls::stream<fixed_t> k_second_stream("k_second_stream");
        hls::stream<fixed_t> qk_part1_stream("qk_part1_stream");
        hls::stream<fixed_t> attention_matrix_stream("attention_matrix_stream");
        hls::stream<fixed_t> attention_softmax_stream("attention_softmax_stream");

#pragma HLS STREAM variable=q_stream depth=64
#pragma HLS STREAM variable=k_stream depth=64
#pragma HLS STREAM variable=q_first_stream depth=64
#pragma HLS STREAM variable=k_first_stream depth=64
#pragma HLS STREAM variable=q_second_stream depth=64
#pragma HLS STREAM variable=k_second_stream depth=64
#pragma HLS STREAM variable=qk_part1_stream depth=N*N
#pragma HLS STREAM variable=attention_matrix_stream depth=64
#pragma HLS STREAM variable=attention_softmax_stream depth=64

        //Load Q and K matrices
        load_qk(Q, K, q_stream, k_stream, b);
        
        //Split the Q and K streams
        split_qk(q_stream, k_stream, q_first_stream, k_first_stream, q_second_stream, k_second_stream);

        //Compute first part of QK^T
        compute_qk_part1(q_first_stream, k_first_stream, qk_part1_stream);
        
        //Compute second part of QK^T and merge
        compute_qk_part2(q_second_stream, k_second_stream, qk_part1_stream, attention_matrix_stream, scale);

        //Apply softmax exactly like original implementation
        softmax_stream(attention_matrix_stream, attention_softmax_stream);

        //Compute output in tiles
        compute_output_tiles(attention_softmax_stream, V, Output, b);
    }
}