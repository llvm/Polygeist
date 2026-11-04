#include "dcl.h"

// Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
void sparse_matrix_multiply_HLS(data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1], 
                                data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1], 
                                data_t C[N][K]) 
{
#pragma HLS interface m_axi port=values_A offset=slave bundle=mem1
#pragma HLS interface m_axi port=column_indices_A offset=slave bundle=mem1
#pragma HLS interface m_axi port=row_ptr_A offset=slave bundle=mem1

#pragma HLS interface m_axi port=values_B offset=slave bundle=mem2
#pragma HLS interface m_axi port=row_indices_B offset=slave bundle=mem2
#pragma HLS interface m_axi port=col_ptr_B offset=slave bundle=mem2

#pragma HLS interface m_axi port=C offset=slave bundle=mem3
#pragma HLS interface s_axilite port=return

    // local buffers for input data
    data_t local_values_A[N * M];
    int local_column_indices_A[N * M];
    int local_row_ptr_A[N + 1];
    
    data_t local_values_B[M * K];
    int local_row_indices_B[M * K];
    int local_col_ptr_B[M + 1];
    
    // copy input data to local buffers

    // copy A values
    for(int i = 0; i < N*M; i++) {
#pragma HLS pipeline
        local_values_A[i] = values_A[i];
    }
    
    // copy A column indices
    for(int i = 0; i < N*M; i++) {
#pragma HLS pipeline
        local_column_indices_A[i] = column_indices_A[i];
    }
    
    // copy A row pointers
    for(int i = 0; i < N+1; i++) {
#pragma HLS pipeline
        local_row_ptr_A[i] = row_ptr_A[i];
    }
    
    // copy B values 
    for(int i = 0; i < M*K; i++) {
#pragma HLS pipeline
        local_values_B[i] = values_B[i];
    }
    
    // copy B row indices
    for(int i = 0; i < M*K; i++) {
#pragma HLS pipeline
        local_row_indices_B[i] = row_indices_B[i];
    }
    
    // copy B column pointers 
    for(int i = 0; i < M+1; i++) {
#pragma HLS pipeline
        local_col_ptr_B[i] = col_ptr_B[i];
    }
    
    // temporary local accumulator for each row of C
    data_t row_accum[K];
#pragma HLS array_partition variable=row_accum complete
    
    // initialize C matrix
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < K; j++) {
#pragma HLS pipeline
            C[i][j] = 0;
        }
    }

    // iterate over rows of A
    for (int i = 0; i < N; i++) {
#pragma HLS pipeline off
        
        // reset accumulator for the current row
        for (int j = 0; j < K; j++) {
#pragma HLS unroll
            row_accum[j] = 0;
        }

        // iterate over nonzero elements of row i in A
        for (int idx_A = local_row_ptr_A[i]; idx_A < local_row_ptr_A[i + 1]; idx_A++) {
#pragma HLS pipeline
            int k = local_column_indices_A[idx_A];
            data_t value_A = local_values_A[idx_A];

            // iterate over nonzero elements in column k of B
            for (int idx_B = local_col_ptr_B[k]; idx_B < local_col_ptr_B[k + 1]; idx_B++) {
#pragma HLS pipeline
                int j = local_row_indices_B[idx_B];
                row_accum[j] += value_A * local_values_B[idx_B];
            }
        }

        // store accumulated row into C
        for (int j = 0; j < K; j++) {
#pragma HLS pipeline
            C[i][j] = row_accum[j];
        }
    }
}