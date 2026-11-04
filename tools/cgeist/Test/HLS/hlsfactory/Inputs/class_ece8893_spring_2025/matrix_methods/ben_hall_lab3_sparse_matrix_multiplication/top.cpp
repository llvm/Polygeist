#include "dcl.h"

// CSR to a 2D matrix
void csr_to_dense(data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1], data_t A[N][M]) {
    // Populate the dense A matrix from CSR format
    for (int i = 0; i < N; i++) {
        for (int idx_A = row_ptr_A[i]; idx_A < row_ptr_A[i + 1]; idx_A++) {
#pragma HLS pipeline
            int col = column_indices_A[idx_A];  // Column index of A
            data_t value = values_A[idx_A];     // Value at A(i, col)
            A[i][col] = value;                  // Store in 2D array
        }
    }
}

void csc_to_dense(data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1], data_t B[M][K]) {
    // Populate the dense B matrix from CSC format
    for (int k = 0; k < M; k++) {
        for (int idx_B = col_ptr_B[k]; idx_B < col_ptr_B[k + 1]; idx_B++) {
#pragma HLS pipeline
            int row = row_indices_B[idx_B];  
            data_t value = values_B[idx_B];  
            B[k][row] = value;
        }
    }
}


void load_data(data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1], 
                             data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1], data_t A[N][M], data_t B[M][K])
{
    #pragma HLS dataflow
    // maybe load in the row indices / col / row ptrs in separately? probably not needed however due to streaming
    csr_to_dense(values_A, column_indices_A, row_ptr_A, A);
    csc_to_dense(values_B, row_indices_B, col_ptr_B, B);
}

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

    // local 2D arrays to store the dense representations of A and B
    data_t A[N][M] = {0};
    data_t B[M][K] = {0};
    load_data(values_A, column_indices_A, row_ptr_A, values_B, row_indices_B, col_ptr_B, A, B);

    data_t C_local[N][K] = {0};
    
    #pragma HLS array_partition variable=C_local dim=2 complete
    
    
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < M; k++) { 
            // split up the multiplication and summation to save DSPs
            data_t to_sum[K] = {0};
            #pragma HLS array_partition variable=to_sum complete
            for (int j = 0; j < K; j++) {
            #pragma HLS unroll
                to_sum[j] = A[i][k] * B[k][j];
            }
            for(int jj = 0; jj < K; jj++) {
            #pragma HLS unroll factor=32
                C_local[i][jj] += to_sum[jj];
            }
        }
    }

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < K; k++) {
            C[i][k] = C_local[i][k];
        }        
    }
}