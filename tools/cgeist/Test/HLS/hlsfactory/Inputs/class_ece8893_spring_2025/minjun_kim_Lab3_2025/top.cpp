#include "dcl.h"
// Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
void sparse_matrix_multiply_HLS(data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1],
                            data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1], data_t C[N][K])
{
    // Memory interface pragmas for efficient memory access
    #pragma HLS interface m_axi port=values_A offset=slave bundle=gmem0
    #pragma HLS interface m_axi port=column_indices_A offset=slave bundle=gmem0
    #pragma HLS interface m_axi port=row_ptr_A offset=slave bundle=gmem0
    #pragma HLS interface m_axi port=values_B offset=slave bundle=gmem1
    #pragma HLS interface m_axi port=row_indices_B offset=slave bundle=gmem1
    #pragma HLS interface m_axi port=col_ptr_B offset=slave bundle=gmem1
    #pragma HLS interface m_axi port=C offset=slave bundle=gmem2
    #pragma HLS interface s_axilite port=return

    // Local result matrix with cyclic partitioning for parallel access
    data_t C_local[N][K];
    #pragma HLS ARRAY_PARTITION variable=C_local cyclic factor=16 dim=2

    // Initialize local result matrix to zero
    for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE
        for (int j = 0; j < K; j++) {
            #pragma HLS UNROLL factor=8
            C_local[i][j] = 0;
        }
    }

    // Perform sparse matrix multiplication
    for (int i = 0; i < N; i++) {
        // Iterate through non-zero elements in current row of A
        for (int idx_A = row_ptr_A[i]; idx_A < row_ptr_A[i + 1]; idx_A++) {
            #pragma HLS PIPELINE II=1
            
            // Get column index and value for current A matrix element
            int k = column_indices_A[idx_A];
            data_t value_A = values_A[idx_A];

            // Iterate through non-zero elements in corresponding column of B
            for (int idx_B = col_ptr_B[k]; idx_B < col_ptr_B[k + 1]; idx_B++) {
                #pragma HLS PIPELINE II=1
                
                // Get column index and value for current B matrix element
                int j = row_indices_B[idx_B];
                data_t value_B = values_B[idx_B];

                // Compute product and accumulate (ignore inter-iteration dependencies)
                #pragma HLS DEPENDENCE variable=C_local inter false
                C_local[i][j] += value_A * value_B;
            }
        }
    }

    // Copy local result matrix to output matrix
    for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE
        for (int j = 0; j < K; j++) {
            #pragma HLS UNROLL factor=8
            C[i][j] = C_local[i][j];
        }
    }
}