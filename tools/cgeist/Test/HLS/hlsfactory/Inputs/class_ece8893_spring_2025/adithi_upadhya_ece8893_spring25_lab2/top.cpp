#include "dcl.h"
#include <string.h>

// Softmax approximation: 
// void softmax_approx_HLS(fixed_t matrix[B][N][N]) {
//     for (int b = 0; b < B; ++b) {
//         for (int i = 0; i < N; ++i) {
//             // Find max value
//             ap_fixed<32, 8> max_val = matrix[b][i][0];
//             #pragma HLS PIPELINE II=1
//             for (int j = 1; j < N; ++j) {
//                 if (matrix[b][i][j] > max_val)
//                   max_val = matrix[b][i][j];
//             }

//             // Approximate exp and sum
//             ap_fixed<32, 8> sum = 0;
//             #pragma HLS PIPELINE II=1
//             for (int j = 0; j < N; ++j) {
//                 // Use a linear approximation of exp for small values
//                 ap_fixed<32, 8> diff = matrix[b][i][j] - max_val;
//                 ap_fixed<32, 8> approximated_exp;

//                 if (diff > (ap_fixed<32,8>)(-5)) {
//                     approximated_exp = ap_fixed<32, 8>(1) + diff + ap_fixed<32, 8>(0.5) * diff * diff;
//                 } else {
//                     approximated_exp = ap_fixed<32,8>(0);
//                 }
//                 matrix[b][i][j] = approximated_exp;
//                 sum += matrix[b][i][j];
//             }

//             // Normalize
//             ap_fixed<32, 8> inv_sum = 1 / sum;
//             #pragma HLS PIPELINE II=1
//             for (int j = 0; j < N; ++j) {
//                 matrix[b][i][j] *= inv_sum;
//             }
//         }
//     }
// }

void softmax_HLS(fixed_t matrix[B][N][N]) {
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            ap_fixed<32, 8> max_val = matrix[b][i][0];
            //adding pipeline pragma was causing impl to fail
            for (int j = 1; j < N; ++j) {
                #pragma HLS UNROLL factor=16 //16 gave the best latency
                if (matrix[b][i][j] > max_val) {
                    max_val = matrix[b][i][j];
                }
            }

            ap_fixed<32, 8> sum = 0;
            for (int j = 0; j < N; ++j) {
                #pragma HLS UNROLL factor=16
                matrix[b][i][j] = hls::exp(matrix[b][i][j] - max_val);
                sum += matrix[b][i][j];
            }

            for (int j = 0; j < N; ++j) {
                #pragma HLS UNROLL factor=16
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

//Local arrays 
fixed_t Q_local[B][N][dk];
fixed_t K_local[B][N][dk];
fixed_t V_local[B][N][dv];

#pragma HLS array_partition variable=Q_local cyclic factor=16
#pragma HLS array_partition variable=K_local cyclic factor=16
#pragma HLS array_partition variable=V_local cyclic factor=16

// Batch-wise copying with burst transfers
// for (int b = 0; b < B; ++b) {
//     #pragma HLS LOOP_FLATTEN
//     memcpy(Q_local[b], Q[b], sizeof(fixed_t) * N * dk);
//     memcpy(K_local[b], K[b], sizeof(fixed_t) * N * dk);
//     memcpy(V_local[b], V[b], sizeof(fixed_t) * N * dv);
// }

// Enforcing Burst mode
for (int b = 0; b < B; ++b) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < dk; ++k) {
            Q_local[b][i][k] = Q[b][i][k]; //copy DRAM array Q
            K_local[b][i][k] = K[b][i][k]; //copy DRAM array K
        }
        for (int j = 0; j < dv; ++j) {
            V_local[b][i][j] = V[b][i][j]; //copy DRAM array V
        }
    }
} 

    fixed_t attention[B][N][N];
    ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);
    
    #pragma HLS array_partition variable=attention cyclic factor=16

    // Compute Q * K^T
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                ap_fixed<32, 8> sum = 0;
                for (int k = 0; k < dk; ++k) {
                    #pragma HLS UNROLL factor=16
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
                    #pragma HLS UNROLL factor=16
                    sum += attention[b][i][k] * V_local[b][k][j];
                }
                Output[b][i][j] = sum;
            }
        }
    }
}