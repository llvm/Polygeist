#include "dcl.h"

// Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
void sparse_matrix_multiply_HLS(data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1],
                                data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1],
                                data_t C[N][K])
{
    // Memory interfaces
#pragma HLS interface m_axi port=values_A offset=slave bundle=mem1
#pragma HLS interface m_axi port=column_indices_A offset=slave bundle=mem1
#pragma HLS interface m_axi port=row_ptr_A offset=slave bundle=mem1

#pragma HLS interface m_axi port=values_B offset=slave bundle=mem2
#pragma HLS interface m_axi port=row_indices_B offset=slave bundle=mem2
#pragma HLS interface m_axi port=col_ptr_B offset=slave bundle=mem2

#pragma HLS interface m_axi port=C offset=slave bundle=mem3
#pragma HLS interface s_axilite port=return

    // Create a local on-chip buffer for C that can be partitioned
    data_t C_local[N][K];
#pragma HLS array_partition variable=C_local complete dim=2

    // Initialize local buffer C_local to zero
    for (int i = 0; i < N; i++) {
    #pragma HLS pipeline II=1
        for (int j = 0; j < K; j++) {
        #pragma HLS unroll factor=4  // Adjust factor based on K and resource availability
            C_local[i][j] = 0;
        }
    }

    // Compute sparse matrix multiplication using the local buffer.
    // Loop over rows of A.
    for (int i = 0; i < N; i++) {
    #pragma HLS pipeline II=1
        // Loop over non-zero elements in row i of A.
        for (int idx_A = row_ptr_A[i]; idx_A < row_ptr_A[i + 1]; idx_A++) {
            int k = column_indices_A[idx_A];  // Column index for A.
            data_t value_A = values_A[idx_A];

            // Loop over non-zero elements in column k of B (CSC format).
            for (int idx_B = col_ptr_B[k]; idx_B < col_ptr_B[k + 1]; idx_B++) {
            #pragma HLS pipeline II=1
            #pragma HLS unroll factor=2  // Unroll inner loop to execute iterations in parallel
                int j = row_indices_B[idx_B];  // Column index for B.
                data_t value_B = values_B[idx_B];

                // Accumulate product into the local buffer.
                C_local[i][j] += value_A * value_B;
            }
        }
    }

    // Write back the computed results from the local buffer to the external memory array C.
    for (int i = 0; i < N; i++) {
    #pragma HLS pipeline II=1
        for (int j = 0; j < K; j++) {
        #pragma HLS unroll factor=4
            C[i][j] = C_local[i][j];
        }
    }
}
