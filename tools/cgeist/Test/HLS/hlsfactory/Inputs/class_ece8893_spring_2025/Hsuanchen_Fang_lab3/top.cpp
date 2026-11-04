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
    // NOTE: Assumes C is pre-initialized to zeros by the caller

    // Local cache for row and column pointers
    int row_ptr_local[N+1];
    int col_ptr_local[M+1];
#pragma HLS ARRAY_PARTITION variable=row_ptr_local complete
#pragma HLS ARRAY_PARTITION variable=col_ptr_local complete
    // Prefetch row and column pointers
    for (int i = 0; i <= N; i++) {
#pragma HLS PIPELINE II=1
        row_ptr_local[i] = row_ptr_A[i];
        if (i <= M) {
            col_ptr_local[i] = col_ptr_B[i];
        }
    }
    // Local buffer for accumulating one row of results
    data_t row_result[K];
#pragma HLS ARRAY_PARTITION variable=row_result cyclic factor=16

    // Perform Sparse x Sparse Multiplication
    for (int i = 0; i < N; i++) {
        // Load current row from C into local buffer
        for (int j = 0; j < K; j++) {
#pragma HLS PIPELINE II=1
            row_result[j] = C[i][j];
        }

        // Cache for storing frequently accessed values and indices
        data_t value_A_cache;
        int col_idx_A_cache;
        int col_start_cache, col_end_cache;

        // Process non-zero elements in row i
        int row_start = row_ptr_local[i];
        int row_end = row_ptr_local[i + 1];

        for (int idx_A = row_start; idx_A < row_end; idx_A++) {
            // Cache the column index and value from A
            col_idx_A_cache = column_indices_A[idx_A];
            value_A_cache = values_A[idx_A];

            // Cache the column pointers from B
            col_start_cache = col_ptr_local[col_idx_A_cache];
            col_end_cache = col_ptr_local[col_idx_A_cache + 1];

            // Process non-zero elements in column col_idx_A_cache of B
            for (int idx_B = col_start_cache; idx_B < col_end_cache; idx_B++) {
#pragma HLS PIPELINE II=1
#pragma HLS dependence variable=row_result inter false
                int j = row_indices_B[idx_B];
                data_t value_B = values_B[idx_B];

                // Accumulate product to local buffer
                row_result[j] += value_A_cache * value_B;
            }
        }

        // Write completed row back to C
        for (int j = 0; j < K; j++) {
#pragma HLS PIPELINE II=1
            C[i][j] = row_result[j];
        }
    }
}