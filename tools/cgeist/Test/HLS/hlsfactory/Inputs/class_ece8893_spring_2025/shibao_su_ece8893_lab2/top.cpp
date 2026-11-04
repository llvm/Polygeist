#include "dcl.h"

// static ap_fixed<16,5> my_exp_poly(ap_fixed<14,4> x) {
//     #pragma HLS INLINE
    
//         //clamp to -2 2
//         if (x < -2) {
//             x = -2;
//         } else if (x > 2) {
//             x = 2;
//         }
//         //    e^x ≈ 1 + x + x^2/2 + x^3/6
//         ap_fixed<16,5> x2 = x * x;
//         ap_fixed<16,5> x3 = x2 * x;
    
//        
//         ap_fixed<16,5> val = (ap_fixed<16,5>)1.0 + x + (x2 / 2) + (x3 / 6);
    
//         return val;
//     }
//didnt really see any improvements
    

void softmax_HLS(fixed_t matrix[B][N][N]) {
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
            ap_fixed<30, 7> max_val = matrix[b][i][0];
            for (int j = 1; j < N; ++j) {
#pragma HLS PIPELINE II=1
                if (matrix[b][i][j] > max_val) {
                    max_val = matrix[b][i][j];
                }
            }

            ap_fixed<30, 7> sum = 0;
            for (int j = 0; j < N; ++j) {
                matrix[b][i][j] = hls::exp(matrix[b][i][j] - max_val);
                //matrix[b][i][j] = my_exp_poly(matrix[b][i][j] - max_val);
#pragma HLS PIPELINE II=1
                sum += matrix[b][i][j];
            }

            for (int j = 0; j < N; ++j) {
#pragma HLS PIPELINE II=1
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


    // Local copies (on-chip memory)
    static fixed_t Q_local[B][N][dk];
    static fixed_t K_local[B][N][dk];
    static fixed_t V_local[B][N][dv];
    static fixed_t Output_local[B][N][dv];
    static fixed_t attention[B][N][N];

//#pragma HLS ARRAY_PARTITION variable=Q_local dim=3 factor=64 block
//#pragma HLS ARRAY_PARTITION variable=K_local dim=3 factor=64 block
//#pragma HLS ARRAY_PARTITION variable=V_local dim=3 factor=64 block
//#pragma HLS ARRAY_PARTITION variable=Output_local dim=3 factor=32 block
//#pragma HLS ARRAY_PARTITION variable=attention dim=3 factor=4 block

//pragma HLS DATAFLOW

// #pragma HLS stream variable=Q_local
// #pragma HLS stream variable=K_local
// #pragma HLS stream variable=V_local
// #pragma HLS stream variable=Output_local
// #pragma HLS stream variable=attention



    // Burst-read Q, K, V from AXI into local arrays
    read_data: for (int b = 0; b < B; b++) {
        for (int i = 0; i < N; i++) {
            for (int d = 0; d < dk; d++) {
#pragma HLS PIPELINE II=1
                Q_local[b][i][d] = Q[b][i][d];
                K_local[b][i][d] = K[b][i][d];
            }
            for (int d = 0; d < dv; d++) {
#pragma HLS PIPELINE II=1
                V_local[b][i][d] = V[b][i][d];
            }
        }
    }


    ap_fixed<30, 7> scale = 1.0 / sqrt((float)dk);

    // Compute Q * K^T
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                ap_fixed<30, 7> sum = 0;
#pragma HLS pipeline II = 1
                for (int k = 0; k < dk; ++k) {
#pragma HLS UNROLL factor = 16
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
#pragma HLS pipeline II = 1              
                ap_fixed<30, 7> sum = 0;
                for (int k = 0; k < N; ++k) {
#pragma HLS UNROLL factor = 16
                    sum += attention[b][i][k] * V_local[b][k][j];
                }
                Output_local[b][i][j] = sum;
            }
        }
    }
//Write results back to Output in AXI
write_data: for (int b = 0; b < B; b++) {
    for (int i = 0; i < N; i++) {
        for (int d = 0; d < dv; d++) {
#pragma HLS PIPELINE II=1
            Output[b][i][d] = Output_local[b][i][d];
        }
    }
}
}


