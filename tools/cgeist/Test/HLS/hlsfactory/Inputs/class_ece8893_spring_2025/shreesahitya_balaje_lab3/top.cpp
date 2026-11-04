// Name: Shree Sahitya Balaje

#include "dcl.h"

// Function to load a CSR (Compressed Sparse Row) matrix into local memory structures
void csr_matrix_loader(
    data_t csr_values[N*M], int col_indices[N*M], int row_ptrs[N+1],
    data_t local_values[N][M], int local_cols[N][M], int local_row_ptrs[N+1])
{
    // Fully partition local row pointers for optimal access
    #pragma HLS array_partition variable=local_row_ptrs complete
    
    // Transfer row pointers from global to local memory
    ROW_PTR_LOAD: for (int r = 0; r <= N; r++) {
        #pragma HLS pipeline II=1
        local_row_ptrs[r] = row_ptrs[r];
    }
    
    // Load non-zero values and column indices for each row
    ROW_LOADER: for (int r = 0; r < N; r++) {
        #pragma HLS pipeline II=1
        int row_begin = local_row_ptrs[r];
        int row_end = local_row_ptrs[r+1];
        
        ELEMENT_LOADER: for (int e = 0; e < row_end - row_begin; e++) {
            #pragma HLS pipeline II=1
            local_values[r][e] = csr_values[row_begin + e];
            local_cols[r][e] = col_indices[row_begin + e];
        }
    }
}

// Function to load a CSC (Compressed Sparse Column) matrix into local memory structures
void csc_matrix_loader(
    data_t csc_values[M*K], int row_indices[M*K], int col_ptrs[M+1],
    data_t local_values[M][K], int local_rows[M][K], int local_col_ptrs[M+1])
{
    // Apply cyclic partitioning for efficient memory access
    #pragma HLS array_partition variable=local_col_ptrs cyclic factor=4
    
    // Transfer column pointers from global to local memory
    COL_PTR_LOAD: for (int c = 0; c <= M; c++) {
        #pragma HLS pipeline II=1
        local_col_ptrs[c] = col_ptrs[c];
    }
    
    // Load non-zero values and row indices for each column
    COL_LOADER: for (int c = 0; c < M; c++) {
        #pragma HLS pipeline II=1
        int col_begin = local_col_ptrs[c];
        int col_end = local_col_ptrs[c+1];
        
        ELEMENT_LOADER: for (int e = 0; e < col_end - col_begin; e++) {
            #pragma HLS pipeline II=1
            local_values[c][e] = csc_values[col_begin + e];
            local_rows[c][e] = row_indices[col_begin + e];
        }
    }
}

// Function to perform sparse matrix multiplication (CSR * CSC)
void matrix_mult_core(
    data_t local_A[N][M], int local_A_cols[N][M], int local_A_rowptrs[N+1],
    data_t local_B[M][K], int local_B_rows[M][K], int local_B_colptrs[M+1],
    data_t result[N][K])
{
    // Apply cyclic partitioning for parallel computation
    #pragma HLS array_partition variable=result cyclic factor=8 dim=2
    
    // Initialize the output matrix with zeros
    RESET_RESULT: for (int i = 0; i < N; i++) {
        #pragma HLS pipeline II=1
        for (int j = 0; j < K; j++) {
            #pragma HLS unroll factor=8
            result[i][j] = 0;
        }
    }

    // Perform matrix multiplication using non-zero elements
    COMPUTE_ROWS: for (int i = 0; i < N; i++) {
        #pragma HLS pipeline II=1
        
        int row_start = local_A_rowptrs[i];
        int row_end = local_A_rowptrs[i+1];
        
        // Traverse non-zero elements in the current row of A
        PROCESS_NONZEROS_A: for (int p = row_start; p < row_end; p++) {
            #pragma HLS pipeline II=1
            int k = local_A_cols[i][p - row_start];
            data_t val_A = local_A[i][p - row_start];
            
            int col_start = local_B_colptrs[k];
            int col_end = local_B_colptrs[k+1];
            
            // Multiply with corresponding non-zero elements in B
            PROCESS_NONZEROS_B: for (int q = col_start; q < col_end; q++) {
                #pragma HLS pipeline II=1
                int j = local_B_rows[k][q - col_start];
                data_t val_B = local_B[k][q - col_start];
                
                result[i][j] += val_A * val_B;
            }
        }
    }
}

// Top-level function for sparse matrix multiplication: A (CSR) * B (CSC) = C (Dense)
void sparse_matrix_multiply_HLS(
    data_t values_A[N*M], int column_indices_A[N*M], int row_ptr_A[N+1],
    data_t values_B[M*K], int row_indices_B[M*K], int col_ptr_B[M+1],
    data_t C[N][K]) 
{
    // Define memory interfaces for efficient data transfer
    #pragma HLS interface m_axi port=values_A offset=slave bundle=mem1
    #pragma HLS interface m_axi port=column_indices_A offset=slave bundle=mem1
    #pragma HLS interface m_axi port=row_ptr_A offset=slave bundle=mem1

    #pragma HLS interface m_axi port=values_B offset=slave bundle=mem2
    #pragma HLS interface m_axi port=row_indices_B offset=slave bundle=mem2
    #pragma HLS interface m_axi port=col_ptr_B offset=slave bundle=mem2

    #pragma HLS interface m_axi port=C offset=slave bundle=mem3
    #pragma HLS interface s_axilite port=return
    
    // Enable HLS dataflow optimization
    #pragma HLS dataflow
    
    // Local memory buffers for storing CSR and CSC representations
    data_t local_A[N][M];
    int local_A_cols[N][M];
    int local_A_rowptrs[N+1];
    
    data_t local_B[M][K];
    int local_B_rows[M][K];
    int local_B_colptrs[M+1];
    
    data_t local_C[N][K];
    
    // Optimize storage for result matrix using BRAM
    #pragma HLS array_partition variable=local_C cyclic factor=8 dim=2
    #pragma HLS bind_storage variable=local_C type=ram_2p impl=bram
    
    // Load matrices from memory into local buffers
    csr_matrix_loader(values_A, column_indices_A, row_ptr_A,
                     local_A, local_A_cols, local_A_rowptrs);
    
    csc_matrix_loader(values_B, row_indices_B, col_ptr_B,
                     local_B, local_B_rows, local_B_colptrs);
    
    // Perform sparse matrix multiplication
    matrix_mult_core(local_A, local_A_cols, local_A_rowptrs,
                   local_B, local_B_rows, local_B_colptrs,
                   local_C);
    
    // Store computed results back into global memory
    STORE_RESULT: for (int i = 0; i < N; i++) {
        #pragma HLS pipeline II=1
        for (int j = 0; j < K; j++) {
            #pragma HLS unroll factor=8
            C[i][j] = local_C[i][j];
        }
    }
}
