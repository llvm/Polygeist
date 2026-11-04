#include "dcl.h"


void softmax_HLS(fixed_t matrix[B][N][N]) {
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            ap_fixed<32, 8> max_val = matrix[b][i][0];
            for (int j = 1; j < N; ++j) {
            // Unrolling the inner loop by 10
            #pragma HLS unroll factor=10
                if (matrix[b][i][j] > max_val) {
                    max_val = matrix[b][i][j];
                }
            }
            ap_fixed<32, 8> sum = 0;
            for (int j = 0; j < N; ++j) {
            // Unrolling the inner loop by 10
            #pragma HLS unroll factor=10
                matrix[b][i][j] = hls::exp(matrix[b][i][j] - max_val);
                sum += matrix[b][i][j];
            }
            // Unrolling the inner loop by 10
            for (int j = 0; j < N; ++j) {
            #pragma HLS unroll factor=10
                matrix[b][i][j] /= sum;
            }
        }
    }
}


// Using wide_t (1024 bits) datatype to fully utilize DRAM AXI bandwidth
void compute_attention_HLS(wide_t Q[B][N][dk/64], wide_t K[B][N][dk/64], wide_t V[B][N][dv/64], wide_t Output[B][N][dv/64]) {
#pragma HLS interface m_axi port=Q offset=slave bundle=mem1
#pragma HLS interface m_axi port=K offset=slave bundle=mem1
#pragma HLS interface m_axi port=V offset=slave bundle=mem1
#pragma HLS interface m_axi port=Output offset=slave bundle=mem2

#pragma HLS interface s_axilite port=return
    
    fixed_t attention[B][N][N];
    // Pre-computing scale factor
    ap_fixed<32, 8> scale = 0.00838835;
    fixed_t Q_local[B][N][dk];
    fixed_t K_local[B][N][dk];
    fixed_t V_local[B][N][dk];
    fixed_t Output_local[B][N][dv];

    // Partitioning the local variables to parallelize (16x) BRAM reads
    // Every variable is partitioned across the third dimension except V_local which
    // is partitioned across the 2nd dimension due to the loop order
    #pragma HLS array_partition variable=Q_local dim=3 factor=8 cyclic
    #pragma HLS array_partition variable=K_local dim=3 factor=8 cyclic
    #pragma HLS array_partition variable=V_local dim=2 factor=10 cyclic
    #pragma HLS array_partition variable=Output_local dim=3 factor=8 cyclic
    #pragma HLS array_partition variable=attention dim=3 factor=10 cyclic

    // Reading wide_t data from DRAM into fixed_t local variables in BRAM
    for (int b = 0; b < B; ++b) {
        for (int n = 0; n < N; ++n) {
            // Pipelining the outer loop
            for (int k = 0; k < dk/64; ++k) {
                // Unrolling the inner loop
                #pragma HLS PIPELINE
                for (int i = 0; i < 64; ++i)
                {
                    // Unrolling the inner loop
                    #pragma HLS unroll
                    Q_local[b][n][k*64+i] = Q[b][n][k/64].range((i+1)*sizeof(fixed_t)-1,i*sizeof(fixed_t));
                    K_local[b][n][k*64+i] = K[b][n][k/64].range((i+1)*sizeof(fixed_t)-1,i*sizeof(fixed_t));
                    V_local[b][n][k*64+i] = V[b][n][k/64].range((i+1)*sizeof(fixed_t)-1,i*sizeof(fixed_t));
                }
            }
        }
    }

    // Compute Q * K^T
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                // Pipelining the outer loop
                ap_fixed<32, 8> sum = 0;                
                for (int k = 0; k < dk; ++k) {
                    // Unrolling the inner loop by 8
                    #pragma HLS unroll factor=8
                    sum += Q_local[b][i][k] * K_local[b][j][k];
                }
                attention[b][i][j] = sum * scale;
            }
        }
    }

    // Apply softmax
    softmax_HLS(attention);

    // Compute Attention * V
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < dv; ++j)
            {
                // Pipelining the outer loop
                ap_fixed<32, 8>  sum = 0;
                for (int k = 0; k < N; ++k) 
                {
                    // Unrolling the inner loop by 8
                    #pragma HLS unroll factor=10
                    sum += attention[b][i][k] * V_local[b][k][j];
                }
                Output_local[b][i][j] = sum;
            }
        }
    }

    // Writing fixed_t local variables to wide_t data in DRAM
    for (int b = 0; b < B; ++b) {
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < dv/64; ++k) {
                // Unrolling the inner loop
                #pragma HLS PIPELINE
                for (int i = 0; i < 64;++i)
                {
                    // Unrolling the inner loop
                    #pragma HLS unroll 
                    Output[b][n][k].range((i+1)*sizeof(fixed_t)-1,i*sizeof(fixed_t)) = Output_local[b][n][k*64+i];
                }
            }
        }
    }    
}
