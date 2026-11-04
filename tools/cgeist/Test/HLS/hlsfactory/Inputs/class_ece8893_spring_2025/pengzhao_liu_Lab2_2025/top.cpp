#include "dcl.h"

void softmax_HLS_local(fixed_t att_local[N][N]) {
    for (int i = 0; i < N; i++) {
        ap_fixed<32, 8> max_val = att_local[i][0];
        for (int j = 1; j < N; j++) {
#pragma HLS pipeline II=1
            if (att_local[i][j] > max_val)
                max_val = att_local[i][j];
        }
        ap_fixed<32, 8> sum = 0;
        for (int j = 0; j < N; j++) {
#pragma HLS pipeline II=1
            att_local[i][j] = 1 + (att_local[i][j] - max_val) + ((att_local[i][j] - max_val) * (att_local[i][j] - max_val)) / 2;
            sum += att_local[i][j];
        }
        for (int j = 0; j < N; j++) {
#pragma HLS pipeline II=1
            att_local[i][j] /= sum;
        }
    }
}


void compute_attention_HLS(fixed_t Q[B][N][dk], fixed_t K[B][N][dk], fixed_t V[B][N][dv], fixed_t Output[B][N][dv]) {
#pragma HLS interface m_axi port=Q offset=slave bundle=mem1
#pragma HLS interface m_axi port=K offset=slave bundle=mem1
#pragma HLS interface m_axi port=V offset=slave bundle=mem1
#pragma HLS interface m_axi port=Output offset=slave bundle=mem2

#pragma HLS interface s_axilite port=return


    // Local buffers for one batch.
    fixed_t Q_buf[N][dk];
    fixed_t K_buf[N][dk];
    fixed_t V_buf[N][dv];
    fixed_t att_buf[N][N];
    fixed_t out_buf[N][dv];

    // Conservative array partitioning:
    // Partition inner dimension of Q and K with factor 4.
#pragma HLS array_partition variable=Q_buf cyclic factor=4 dim=2
#pragma HLS array_partition variable=K_buf cyclic factor=4 dim=2
    // Partition inner dimension of V, att_buf and out_buf with factor 2.
#pragma HLS array_partition variable=V_buf cyclic factor=2 dim=2
#pragma HLS array_partition variable=att_buf cyclic factor=2 dim=2
#pragma HLS array_partition variable=out_buf cyclic factor=2 dim=2

    for (int b = 0; b < B; b++) {
// #pragma HLS DATAFLOW
        // 1) Burst-load Q and K for batch b.
        for (int i = 0; i < N; i++) {
            for (int x = 0; x < dk; x++) {
#pragma HLS pipeline II=8
                Q_buf[i][x] = Q[b][i][x];
                K_buf[i][x] = K[b][i][x];
            }
        }
        // 2) Burst-load V.
        for (int i = 0; i < N; i++) {
            for (int x = 0; x < dv; x++) {
#pragma HLS pipeline II=8
                V_buf[i][x] = V[b][i][x];
            }
        }
        // 3) Compute Q*K^T (scaled) -> att_buf.
        ap_fixed<32,8> scale = 1.0 / sqrt((float)dk);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
#pragma HLS pipeline II=8
                ap_fixed<32,8> sum = 0;
                for (int x = 0; x < dk; x++) {
#pragma HLS unroll factor=2
                    sum += Q_buf[i][x] * K_buf[j][x];
                }
                att_buf[i][j] = sum * scale;
            }
        }
        // 4) Softmax.
        softmax_HLS_local(att_buf);

        // 5) Multiply the attention matrix with V_buf -> out_buf.
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < dv; j++) {
#pragma HLS pipeline II=8
                ap_fixed<32,8> sum = 0;
                for (int x = 0; x < N; x++) {
#pragma HLS unroll factor=2
                    sum += att_buf[i][x] * V_buf[x][j];
                }
                out_buf[i][j] = sum;
            }
        }
        // 6) Burst-store the output back to DRAM.
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < dv; j++) {
#pragma HLS pipeline II=8
                Output[b][i][j] = out_buf[i][j];
            }
        }
    }
}