#include "dcl.h"

void softmax_HLS(fixed_t matrix[B][N][N]) {
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            //reducing the number of bits to 12
            ap_fixed<12, 8> max_val = matrix[b][i][0];
            for (int j = 1; j < N; ++j) {
                if (matrix[b][i][j] > max_val) {
                    max_val = matrix[b][i][j];
                }
            }

            //reducing the number of bits to 12
            ap_fixed<12, 8> sum = 0;
            for (int j = 0; j < N; ++j) {
                matrix[b][i][j] = hls::exp(matrix[b][i][j] - max_val);
                sum += matrix[b][i][j];
            }

            for (int j = 0; j < N; ++j) {
                matrix[b][i][j] /= sum;
            }
        }
    }
}


void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], fixed_t Output[B][N][dv]) {
#pragma HLS interface m_axi port=Q offset=slave bundle=mem1
#pragma HLS interface m_axi port=K offset=slave bundle=mem1
#pragma HLS interface m_axi port=V offset=slave bundle=mem1
#pragma HLS interface m_axi port=Output offset=slave bundle=mem2

#pragma HLS interface s_axilite port=return

//local memory to store in BRAMs
fixed_t Q_local[B][N][dk];
fixed_t K_local[B][N][dk];
fixed_t V_local[B][N][dv];
fixed_t Output_local[B][N][dv];
fixed_t attention[B][N][N];

//array partitioning
#pragma HLS array_partition variable=Q_local dim=1 complete
#pragma HLS array_partition variable=K_local dim=1 complete
#pragma HLS array_partition variable=V_local dim=1 complete
#pragma HLS array_partition variable=Output_local dim=1 complete
#pragma HLS array_partition variable=attention dim=1 complete

//reducing the number of bits to 12
ap_fixed<12, 8> scale = 1.0 / sqrt((float)dk);

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < dk; ++k) {
                Q_local[b][i][k] = Q[b][i][k];
                K_local[b][i][k] = K[b][i][k];
            }
            for (int v = 0; v < dv; ++v) {
                V_local[b][i][v] = V[b][i][v];
            }
        }
    }

    // Compute Q * K^T
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                ap_fixed<32, 8> sum = 0;
                for (int k = 0; k < dk; ++k) {
                    //parallel computation using unrolling
                    #pragma HLS unroll factor=32
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
            for (int j = 0; j < dv; ++j) {
                ap_fixed<32, 8> sum = 0;
                for (int k = 0; k < N; ++k) {
                    //parallel computation using unrolling
                    #pragma HLS unroll factor=32
                    sum += attention[b][i][k] * V_local[b][k][j];
                }
                Output_local[b][i][j] = sum;
            }
        }
    }

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < dv; ++j) {
                //paralell loading of output
                #pragma HLS unroll factor=32
                Output[b][i][j] = Output_local[b][i][j];
            }
        }
    }
}