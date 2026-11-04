#include "dcl.h"

void softmax_HLS(fixed_t matrix[N][N]) {
    for (int i = 0; i < N; ++i) {
        Compute_max:
        ap_fixed<32, 8> max_val = matrix[i][0];
        for (int j = 0; j < N; j += fn) {
        #pragma HLS pipeline II=1
            for (int jj = 0; jj < fn; ++jj){
            #pragma HLS unroll
                if (matrix[i][j + jj] > max_val) {
                    max_val = matrix[i][j + jj];
                }
            }
        }

        Compute_exp:
        ap_fixed<32, 8> sum = 0;
        for (int j = 0; j < N; ++j) {
        #pragma HLS unroll factor=fn
            matrix[i][j] = hls::exp(matrix[i][j] - max_val);
            sum += matrix[i][j];
        }

        compute_matrix:
        for (int j = 0; j < N; ++j) {
        #pragma HLS unroll factor=fn
            matrix[i][j] /= sum;
        }
    }
}

void copy_to_bram(fixed_t Q[N][dk], fixed_t K[N][dk], fixed_t V[N][dv],
                  fixed_t Q_local[N][dk], fixed_t K_local[N][dk], fixed_t V_local[N][dv]) {
    CopyQ: for (int i = 0; i < N; ++i) {
        for (int k = 0; k < dk; ++k) {
            Q_local[i][k] = Q[i][k];
        }
    }
    
    CopyK: for (int i = 0; i < N; ++i) {
        for (int k = 0; k < dk; ++k) {
            K_local[i][k] = K[i][k];
        }
    }
    
    CopyV: for (int i = 0; i < N; ++i) {
        for (int k = 0; k < dv; ++k) {
            V_local[i][k] = V[i][k];
        }
    }
}

void compute_attention_HLS_in(fixed_t Q[N][dk], fixed_t K[N][dk], fixed_t V[N][dv], fixed_t Output[N][dv]) {
#pragma HLS INLINE OFF
    fixed_t Q_local[N][dk];
    fixed_t K_local[N][dk];
    fixed_t V_local[N][dv];
    fixed_t attention[N][N];

#pragma HLS array_partition variable=Q_local dim=2 factor=fd cyclic
#pragma HLS array_partition variable=K_local dim=2 factor=fd cyclic
#pragma HLS array_partition variable=V_local dim=1 factor=fd cyclic

copy_to_bram(Q, K, V, Q_local, K_local, V_local);

ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);

// Compute Q * K^T
ComputeQKT:
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            ap_fixed<32, 8> sum = 0;
            for (int k = 0; k < dk; k += fd) {
            #pragma HLS pipeline II=1
                for (int kk = 0; kk < fd; ++kk) {
                #pragma HLS unroll
                    sum += Q_local[i][k + kk] * K_local[j][k + kk];
                }
            }
            attention[i][j] = sum * scale;
        }
    }

// Apply softmax
softmax_HLS(attention);

// Compute Attention * V
ComupteAttention:
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < dv; ++j) {
            ap_fixed<32, 8> sum = 0;
            for (int k = 0; k < N; ++k) {
                #pragma HLS unroll
                    sum += attention[i][k] * V_local[k][j];
                }
            Output[i][j] = sum;
        }
    }
}

// 32batch
void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], fixed_t Output[B][N][dv]) {
#pragma HLS interface m_axi port=Q offset=slave bundle=mem1
#pragma HLS interface m_axi port=K offset=slave bundle=mem2
#pragma HLS interface m_axi port=V offset=slave bundle=mem3
#pragma HLS interface m_axi port=Output offset=slave bundle=mem4
#pragma HLS interface s_axilite port=return

    for (int b = 0; b < B; ++b) {
    #pragma HLS dataflow
        compute_attention_HLS_in(Q[b], K[b], V[b], Output[b]);
    }
}

