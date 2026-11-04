#include "dcl.h"

// Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
void sparse_matrix_multiply_HLS(data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1], 
                                data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1], 
                                data_t C[N][K]) 
{
    // AXI Interface Declarations for DRAM ports
#pragma HLS interface m_axi port=values_A offset=slave bundle=mem1
#pragma HLS interface m_axi port=column_indices_A offset=slave bundle=mem1
#pragma HLS interface m_axi port=row_ptr_A offset=slave bundle=mem1

#pragma HLS interface m_axi port=values_B offset=slave bundle=mem2
#pragma HLS interface m_axi port=row_indices_B offset=slave bundle=mem2
#pragma HLS interface m_axi port=col_ptr_B offset=slave bundle=mem2

#pragma HLS interface m_axi port=C offset=slave bundle=mem3
#pragma HLS interface s_axilite port=return

    // Local BRAM Buffers for Matrix A (CSR format)
    data_t values_A_local[N * M];                // Non-zero values of A
    int column_indices_A_local[N * M];           // Column indices for each non-zero in A
    int row_ptr_A_local[N + 1];                  // Row pointer array for A

    // Local BRAM Buffers for Matrix B (CSC format)
    data_t values_B_local[M * K];                // Non-zero values of B
    int row_indices_B_local[M * K];              // Row indices for each non-zero in B
    int col_ptr_B_local[M + 1];                  // Column pointer array for B

    // Local accumulation buffer for the dense result matrix C
    data_t C_local[N][K];

    // Copy Matrix A's data from DRAM to local BRAM
    for (int i = 0; i < N * M; i++) {
        values_A_local[i] = values_A[i];
        column_indices_A_local[i] = column_indices_A[i];
    }

    // Copy row pointer array of A
    for (int i = 0; i < N + 1; i++) {
        row_ptr_A_local[i] = row_ptr_A[i];
    }

    // Copy Matrix B's data from DRAM to local BRAM
    for (int i = 0; i < M * K; i++) {
        values_B_local[i] = values_B[i];
        row_indices_B_local[i] = row_indices_B[i];
    }

    // Copy column pointer array of B with pipelining enabled for better performance
    for (int i = 0; i < M + 1; i++) {
        #pragma HLS pipeline
        col_ptr_B_local[i] = col_ptr_B[i];
    }

    // Initialize the local result matrix to zero
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            C_local[i][j] = 0;
        }
    }

    // Perform Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
    for (int i = 0; i < N; i++) {
        // Loop through non-zero entries in row i of A
        for (int idx_A = row_ptr_A_local[i]; idx_A < row_ptr_A_local[i + 1]; idx_A++) {
            int k = column_indices_A_local[idx_A];  // Column index in A corresponds to row in B
            data_t value_A = values_A_local[idx_A]; // Non-zero value from A

            // Loop through non-zero entries in column k of B
            for (int idx_B = col_ptr_B_local[k]; idx_B < col_ptr_B_local[k + 1]; idx_B++) {
                int j = row_indices_B_local[idx_B];  // Column index in result matrix C
                data_t value_B = values_B_local[idx_B]; // Non-zero value from B

                // Accumulate product into the output matrix
                C_local[i][j] += value_A * value_B;
            }
        }
    }

    // Write final result from local C buffer back to DRAM
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            C[i][j] = C_local[i][j];
        }
    }
}
