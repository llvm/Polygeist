#include "dcl.h"

// Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
void sparse_matrix_multiply_HLS(
    wide_t values_A[N * M / 32], int column_indices_A[N * M], int row_ptr_A[N + 1],
    wide_t values_B[M * K / 32], int row_indices_B[M * K], int col_ptr_B[M + 1],
    data_t C[N][K])
{
#pragma HLS INTERFACE m_axi port=values_A offset=slave bundle=mem1 depth=4096
#pragma HLS INTERFACE m_axi port=column_indices_A offset=slave bundle=mem1 depth=4096
#pragma HLS INTERFACE m_axi port=row_ptr_A offset=slave bundle=mem1 depth=128

#pragma HLS INTERFACE m_axi port=values_B offset=slave bundle=mem2 depth=4096
#pragma HLS INTERFACE m_axi port=row_indices_B offset=slave bundle=mem2 depth=4096
#pragma HLS INTERFACE m_axi port=col_ptr_B offset=slave bundle=mem2 depth=128

#pragma HLS INTERFACE m_axi port=C offset=slave bundle=mem3 depth=4096
#pragma HLS INTERFACE s_axilite port=return

    // Local buffer for A and B (unpacked from wide_t)
    data_t values_A_local[N * M];
    data_t values_B_local[M * K];
    #pragma HLS ARRAY_PARTITION variable=values_A_local cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=values_B_local cyclic factor=8 dim=1

    // Local accumulation buffer
    data_t C_local[N][K];
    #pragma HLS ARRAY_PARTITION variable=C_local cyclic factor=8 dim=2

    // Unpack wide values_A into local buffer
    for (int i = 0; i < (N * M) / 32; ++i) {
        #pragma HLS PIPELINE
        wide_t wide_val = values_A[i];
        for (int j = 0; j < 32; ++j) {
            #pragma HLS UNROLL
            values_A_local[i * 32 + j] = *((data_t*)&wide_val + j);
        }
    }

    // Unpack wide values_B into local buffer
    for (int i = 0; i < (M * K) / 32; ++i) {
        #pragma HLS PIPELINE
        wide_t wide_val = values_B[i];
        for (int j = 0; j < 32; ++j) {
            #pragma HLS UNROLL
            values_B_local[i * 32 + j] = *((data_t*)&wide_val + j);
        }
    }

    // Initialize output buffer
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            #pragma HLS PIPELINE
            C_local[i][j] = 0;
        }
    }

    // Sparse matrix multiplication
    for (int i = 0; i < N; i++) {
        int row_start = row_ptr_A[i];
        int row_end = row_ptr_A[i + 1];

        for (int idx_A = row_start; idx_A < row_end; idx_A++) {
            #pragma HLS PIPELINE II=1
            int k = column_indices_A[idx_A];     // Column of A = row of B
            data_t value_A = values_A_local[idx_A];

            int col_start = col_ptr_B[k];
            int col_end = col_ptr_B[k + 1];

            for (int idx_B = col_start; idx_B < col_end; idx_B++) {
                #pragma HLS UNROLL factor=4
                int j = row_indices_B[idx_B];     // Column of result
                data_t value_B = values_B_local[idx_B];
                C_local[i][j] += value_A * value_B;
            }
        }
    }

    // Write back to DRAM
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            #pragma HLS PIPELINE
            C[i][j] = C_local[i][j];
        }
    }
}