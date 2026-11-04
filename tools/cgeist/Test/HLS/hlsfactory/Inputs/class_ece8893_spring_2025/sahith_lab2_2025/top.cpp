#include "dcl.h"



void softmax_HLS(fixed_t matrix[B][N][N]) {
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            ap_fixed<32, 8> max_val = matrix[b][i][0];
            for (int j = 1; j < N; ++j) {
				#pragma HLS unroll factor=8
                max_val = (matrix[b][i][j] > max_val) ? matrix[b][i][j] : max_val;
            }

            ap_fixed<32, 8> sum = 0;
            for (int j = 0; j < N; ++j) {
				#pragma HLS unroll factor=8
                matrix[b][i][j] = hls::exp(matrix[b][i][j] - max_val);
                sum += matrix[b][i][j];
            }

            for (int j = 0; j < N; ++j) {
				#pragma HLS unroll factor=8
                matrix[b][i][j] /= sum;
            }
        }
    }
}

void compute_QKT(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t attention[B][N][N]){
    ap_fixed<32, 8> scale = 1.0 / sqrt((float)dk);
	#pragma HLS array_partition variable=Q complete dim=3
	#pragma HLS array_partition variable=K complete dim=3

    // Compute Q * K^T
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
				#pragma HLS pipeline II=1

                ap_fixed<32, 8> sum = 0;
                for (int k = 0; k < dk; ++k) {
					#pragma HLS unroll factor=8
                    sum += Q[b][i][k] * K[b][j][k];
                }
                attention[b][i][j] = sum * scale;
            }
        }
    }
}

void compute_V(fixed_t attention[B][N][N], fixed_t V[B][N][dv], fixed_t Output[B][N][dv]) {
	#pragma HLS array_partition variable=attention complete dim=3
	#pragma HLS array_partition variable=V complete dim=2
    // Compute Attention * V
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < dv; ++j) {
				#pragma HLS pipeline II=1
                ap_fixed<32, 8> sum = 0;
                for (int k = 0; k < N; ++k) {
					#pragma HLS unroll factor=8
                    sum += attention[b][i][k] * V[b][k][j];
                }
                Output[b][i][j] = sum;
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


	fixed_t attention[B][N][N], Q_local[B][N][dk], K_local[B][N][dk], V_local[B][N][dv];

	for (int b = 0; b<B; b++){
		for (int i=0; i<N; i++){
			for (int j=0; j<dk; j++){
				Q_local[b][i][j] = Q[b][i][j];
			}
			for (int j=0; j<dk; j++){
				K_local[b][i][j] = K[b][i][j];
			}
			for (int j=0; j<dv; j++){
				V_local[b][i][j] = V[b][i][j];
			}
		}
	}

	compute_QKT(Q_local, K_local, attention);
    softmax_HLS(attention);
	compute_V(attention, V_local, Output);
}
