#include "dcl.h"

// Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
void sparse_matrix_multiply_HLS(data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1], 
                             data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1], data_t C[N][K]) 
{
#pragma HLS interface m_axi port=values_A offset=slave bundle=mem1
#pragma HLS interface m_axi port=column_indices_A offset=slave bundle=mem1
#pragma HLS interface m_axi port=row_ptr_A offset=slave bundle=mem1

#pragma HLS interface m_axi port=values_B offset=slave bundle=mem2
#pragma HLS interface m_axi port=row_indices_B offset=slave bundle=mem2
#pragma HLS interface m_axi port=col_ptr_B offset=slave bundle=mem2

#pragma HLS interface m_axi port=C offset=slave bundle=mem3

#pragma HLS interface s_axilite port=return

// BRAM Buffers
    data_t local_values_A[N * M];
    int local_column_indices_A[N * M];
    int local_row_ptr_A[N + 1];

    data_t local_values_B[M * K];
    int local_row_indices_B[M * K];
    int local_col_ptr_B[M + 1];

    data_t local_C[N][K];

    // Initialize local C to zero
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            local_C[i][j] = 0;
        }
    }

    // Copy A to local BRAM
    for (int i = 0; i < N * M; i++) {
        local_values_A[i] = values_A[i];
        local_column_indices_A[i] = column_indices_A[i];
    }
    for (int i = 0; i < N + 1; i++) {
        local_row_ptr_A[i] = row_ptr_A[i];
    }

    // Copy B to local BRAM
    for (int i = 0; i < M * K; i++) {
        local_values_B[i] = values_B[i];
        local_row_indices_B[i] = row_indices_B[i];
    }
    for (int i = 0; i < M + 1; i++) {
        local_col_ptr_B[i] = col_ptr_B[i];
    }

    // Perform Sparse x Sparse Multiplication
    for (int i = 0; i < N; i++) {
        for (int idx_A = local_row_ptr_A[i]; idx_A < local_row_ptr_A[i + 1]; idx_A++) {
            int k = local_column_indices_A[idx_A]; // Column index of A
            data_t value_A = local_values_A[idx_A];

            int len_B = local_col_ptr_B[k + 1] - local_col_ptr_B[k];
            int idx_B = local_col_ptr_B[k];

            // Unrolling for efficient computation (Factor 16)
            if (len_B >= 16) {
                for (; idx_B + 15 < local_col_ptr_B[k + 1]; idx_B += 16) {
                    #pragma HLS unroll

                    for (int u = 0; u < 16; u++) {
                        local_C[i][local_row_indices_B[idx_B + u]] += value_A * local_values_B[idx_B + u];
                    }
                }
            }

            // Handle any remaining iterations (optimized)
            for (; idx_B < local_col_ptr_B[k + 1]; idx_B++) {
                local_C[i][local_row_indices_B[idx_B]] += value_A * local_values_B[idx_B];
            }
        }
    }

    // Write local_C back to DRAM
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            C[i][j] = local_C[i][j];
        }
    }
}
