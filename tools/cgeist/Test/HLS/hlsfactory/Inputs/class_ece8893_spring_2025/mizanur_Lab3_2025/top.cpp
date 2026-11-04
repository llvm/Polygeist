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

#pragma HLS DATAFLOW

    // Local BRAM Buffers
    data_t local_C[N][K];
    #pragma HLS ARRAY_PARTITION variable=local_C complete dim=2

    // Initialize local buffer
    init_C: for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            #pragma HLS PIPELINE II=1
            local_C[i][j] = 0;
        }
    }

    //load metadata and values on BRAM
    // Partition metadata and values for improved parallel access


    data_t values_A_loc[N * M];
    data_t values_B_loc[M * K];
    int column_indices_A_loc[N * M];
    int row_indices_B_loc[M * K];
    int row_ptr_A_loc[N + 1];
    int col_ptr_B_loc[M + 1];

    #pragma HLS ARRAY_PARTITION variable=values_A_loc cyclic factor=64 dim=1
    #pragma HLS ARRAY_PARTITION variable=column_indices_A_loc cyclic factor=64 dim=1
    #pragma HLS ARRAY_PARTITION variable=row_indices_B_loc cyclic factor=64 dim=1
    #pragma HLS ARRAY_PARTITION variable=values_B_loc cyclic factor=64 dim=1

    #pragma HLS ARRAY_PARTITION variable=row_ptr_A_loc complete dim=1
    #pragma HLS ARRAY_PARTITION variable=col_ptr_B_loc complete dim=1

    // load_A: for (int i = 0; i < N * M; i++) {
    //     #pragma HLS PIPELINE II=1
    //     values_A_loc[i] = values_A[i];
    //     column_indices_A_loc[i] = column_indices_A[i];
    // }

    memcpy(values_A_loc, values_A, sizeof(data_t) * N * M);

    memcpy(values_B_loc, values_B, sizeof(data_t) * M * K);

    load_row_ptr_A: for (int i = 0; i < N + 1; i++) {
        #pragma HLS PIPELINE II=1
        row_ptr_A_loc[i] = row_ptr_A[i];
    }

    // load_B: for (int i = 0; i < M * K; i++) {
    //     #pragma HLS PIPELINE II=1
    //     values_B_loc[i] = values_B[i];
    //     row_indices_B_loc[i] = row_indices_B[i];
    // }

    load_col_ptr_B: for (int i = 0; i < M + 1; i++) {
        #pragma HLS PIPELINE II=1
        col_ptr_B_loc[i] = col_ptr_B[i];
    }

    // Perform Sparse x Sparse Multiplication
    row_loop: for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL FACTOR=4
        
        col_A_loop: for (int idx_A = row_ptr_A_loc[i]; idx_A < row_ptr_A_loc[i + 1]; idx_A++) {
            #pragma HLS PIPELINE II=1
            
            // Iterate over columns of B corresponding to row k
            data_t valA = values_A_loc[idx_A];
            col_B_loop: for (int idx_B = col_ptr_B_loc[column_indices_A_loc[idx_A]]; idx_B < col_ptr_B_loc[column_indices_A_loc[idx_A] + 1]; idx_B++) {
                #pragma HLS PIPELINE II=1 

                // Accumulate the product into local_C[i][j]
                local_C[i][row_indices_B_loc[idx_B]] += valA * values_B_loc[idx_B];
            }
        }
    }

    // Write back to global memory
    write_C: for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            #pragma HLS PIPELINE II=1
            C[i][j] = local_C[i][j];
        }
    }
}
