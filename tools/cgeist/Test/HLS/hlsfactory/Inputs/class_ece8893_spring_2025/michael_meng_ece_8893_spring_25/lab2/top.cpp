#include "dcl.h"



void softmax_HLS(fixed_t matrix[B][N][N]) {
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            ap_fixed<32, 8> max_val = matrix[b][i][0];
            for (int j = 1; j < N; ++j) {
                if (matrix[b][i][j] > max_val) {
                    max_val = matrix[b][i][j];
                }
            }

            ap_fixed<32, 8> sum = 0;
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


fixed_t attention[B][N][N];
    ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);

    // Compute Q * K^T
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                ap_fixed<32, 8> sum = 0;
                for (int k = 0; k < dk; ++k) {
                    sum += Q[b][i][k] * K[b][j][k];
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
                    sum += attention[b][i][k] * V[b][k][j];
                }
                Output[b][i][j] = sum;
            }
        }
    }


}
