//ORIGINAL
//#include "dcl.h"
//
//
//
//void softmax_HLS(fixed_t matrix[B][N][N]) {
//    for (int b = 0; b < B; ++b) {
//        for (int i = 0; i < N; ++i) {
//            ap_fixed<32, 8> max_val = matrix[b][i][0];
//            for (int j = 1; j < N; ++j) {
//                if (matrix[b][i][j] > max_val) {
//                    max_val = matrix[b][i][j];
//                }
//            }
//
//            ap_fixed<32, 8> sum = 0;
//            for (int j = 0; j < N; ++j) {
//                matrix[b][i][j] = hls::exp(matrix[b][i][j] - max_val);
//                sum += matrix[b][i][j];
//            }
//
//            for (int j = 0; j < N; ++j) {
//                matrix[b][i][j] /= sum;
//            }
//        }
//    }
//}
//
//
//void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], fixed_t Output[B][N][dv]) {
//#pragma HLS interface m_axi port=Q offset=slave bundle=mem1
//#pragma HLS interface m_axi port=K offset=slave bundle=mem1
//#pragma HLS interface m_axi port=V offset=slave bundle=mem1
//#pragma HLS interface m_axi port=Output offset=slave bundle=mem2
//
//#pragma HLS interface s_axilite port=return
//
//
//fixed_t attention[B][N][N];
//    ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);
//
//    // Compute Q * K^T
//    for (int b = 0; b < B; ++b) {
//        for (int i = 0; i < N; ++i) {
//            for (int j = 0; j < N; ++j) {
//                ap_fixed<32, 8> sum = 0;
//                for (int k = 0; k < dk; ++k) {
//                    sum += Q[b][i][k] * K[b][j][k];
//                }
//                attention[b][i][j] = sum * scale;
//            }
//        }
//    }
//
//    // Apply softmax
//    softmax_HLS(attention);
//
//    // Compute Attention * V
//    for (int b = 0; b < B; ++b) {
//        for (int i = 0; i < N; ++i) {
//            for (int j = 0; j < dv; ++j) {
//                ap_fixed<32, 8> sum = 0;
//                for (int k = 0; k < N; ++k) {
//                    sum += attention[b][i][k] * V[b][k][j];
//                }
//                Output[b][i][j] = sum;
//            }
//        }
//    }
//
//
//}


// OPTIMIZED

#include "dcl.h"

void softmax_HLS(fixed_t matrix[N][N]) {
    #pragma HLS INLINE
    for (int i = 0; i < N; ++i) {
        ap_fixed<32, 8> max_val = -1e9;
        ap_fixed<32, 8> sum = 0;
        ap_fixed<32, 8> temp[N];

        #pragma HLS ARRAY_PARTITION variable=temp cyclic factor=8 dim=1

        // Find max value
        for (int j = 0; j < N; ++j) {
            #pragma HLS PIPELINE II=1
            max_val = (ap_fixed<32, 8>)matrix[i][j] > max_val ? (ap_fixed<32, 8>)matrix[i][j] : max_val;
        }

        // Compute exponentials & sum
        for (int j = 0; j < N; ++j) {
            #pragma HLS PIPELINE II=1
            temp[j] = hls::exp(matrix[i][j] - max_val);
            sum += temp[j];
        }

        // Normalize
        for (int j = 0; j < N; ++j) {
            #pragma HLS PIPELINE II=1
            matrix[i][j] = temp[j] / sum;
        }
    }
}


void compute_attention_HLS(fixed_t Q[B][N][dk],fixed_t K[B][N][dk],fixed_t V[B][N][dv],fixed_t Output[B][N][dv]) {
#pragma HLS INTERFACE m_axi port=Q offset=slave bundle=mem1
#pragma HLS INTERFACE m_axi port=K offset=slave bundle=mem1
#pragma HLS INTERFACE m_axi port=V offset=slave bundle=mem1
#pragma HLS INTERFACE m_axi port=Output offset=slave bundle=mem2
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS DATAFLOW 


    fixed_t Q_local[N][dk];
    fixed_t K_local[N][dk];
    fixed_t V_local[N][dv];
    fixed_t attention[N][N];

#pragma HLS ARRAY_PARTITION variable=Q_local cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=K_local cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=V_local cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=attention cyclic factor=8 dim=2

    ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);

    
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < dk; ++k) {
                #pragma HLS PIPELINE II=1
                Q_local[i][k] = Q[b][i][k];
                K_local[i][k] = K[b][i][k];
            }
            for (int k = 0; k < dv; ++k) {
                  #pragma HLS PIPELINE II=1
                V_local[i][k] = V[b][i][k];
            }
        }

        // Compute Q * K^T
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                ap_fixed<32, 8> sum = 0;
                  #pragma HLS PIPELINE II=1
                  #pragma HLS UNROLL factor=8
                for (int k = 0; k < dk; ++k) {
                    sum += Q_local[i][k] * K_local[j][k];
                }
                attention[i][j] = sum * scale;
            }
        }

        // Apply softmax
        softmax_HLS(attention);

        // Compute Attention * V
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < dv; ++j) {
                ap_fixed<32, 8> sum = 0;
                #pragma HLS PIPELINE II=1
                #pragma HLS UNROLL factor=8
                for (int k = 0; k < N; ++k) {
                    sum += attention[i][k] * V_local[k][j];
                }
                Output[b][i][j] = sum;
            }
        }
    }
}





