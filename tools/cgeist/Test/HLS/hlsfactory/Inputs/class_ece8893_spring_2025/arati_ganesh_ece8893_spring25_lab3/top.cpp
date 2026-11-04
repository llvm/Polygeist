#include "dcl.h"


void sparse_matrix_multiply_HLS(data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1], data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1],
  data_t C[N][K])
  {
    // DRAM interface declarations.
    #pragma HLS interface m_axi port=values_A offset=slave bundle=mem1
    #pragma HLS interface m_axi port=column_indices_A offset=slave bundle=mem1
    #pragma HLS interface m_axi port=row_ptr_A offset=slave bundle=mem1
    #pragma HLS interface m_axi port=values_B offset=slave bundle=mem2
    #pragma HLS interface m_axi port=row_indices_B offset=slave bundle=mem2
    #pragma HLS interface m_axi port=col_ptr_B offset=slave bundle=mem2
    #pragma HLS interface m_axi port=C offset=slave bundle=mem3
    #pragma HLS interface s_axilite port=return
    
    data_t local_values_A[N * M];
    #pragma HLS ARRAY_PARTITION variable=local_values_A block factor=4 dim=1
    
    int local_column_indices_A[N * M];
    #pragma HLS ARRAY_PARTITION variable=local_column_indices_A cyclic factor=4 dim=1
    
    int local_row_ptr_A[N + 1];
    #pragma HLS ARRAY_PARTITION variable=local_row_ptr_A complete dim=1
    
    data_t local_values_B[M * K];
    #pragma HLS ARRAY_PARTITION variable=local_values_B block factor=4 dim=1
    
    int local_row_indices_B[M * K];
    #pragma HLS ARRAY_PARTITION variable=local_row_indices_B cyclic factor=4 dim=1
    
    int local_col_ptr_B[M + 1];
    #pragma HLS ARRAY_PARTITION variable=local_col_ptr_B complete dim=1
    
    // Create a local buffer for C matrix
    data_t local_C[N][K];
    #pragma HLS ARRAY_PARTITION variable=local_C complete dim=2
    
    // Load input data from DRAM to local memory
    for (int i = 0; i < N * M; i++) {
        #pragma HLS PIPELINE II=1
        local_values_A[i] = values_A[i];
        local_column_indices_A[i] = column_indices_A[i];
    }
    
    for (int i = 0; i < N + 1; i++) {
    #pragma HLS PIPELINE II=1
        local_row_ptr_A[i] = row_ptr_A[i];
    }
    
    for (int i = 0; i < M * K; i++) {
    #pragma HLS PIPELINE II=1
        local_values_B[i] = values_B[i];
        local_row_indices_B[i] = row_indices_B[i];
    }
    
    for (int i = 0; i < M + 1; i++) {
    #pragma HLS PIPELINE II=1
        local_col_ptr_B[i] = col_ptr_B[i];
    }

  
    const int S = 4;
    //Partial Unroll for variable loop bounds
    // for (int i = 0; i < N; i++) {
    //     #pragma HLS PIPELINE II=1
    
    //     // Iterate over nonzeros in row i of A.
    //     for (int idx_A = local_row_ptr_A[i]; idx_A < local_row_ptr_A[i + 1]; idx_A++) {
    //         #pragma HLS LOOP_TRIPCOUNT min=1 max=64 avg=16
    //         int k = local_column_indices_A[idx_A];
    //         data_t value_A = local_values_A[idx_A];
            
    //         // Multiply value_A with the k-th column of B in CSC format.
    //         for (int idx_B = local_col_ptr_B[k]; idx_B < local_col_ptr_B[k + 1]; idx_B += S) {
    //             #pragma HLS PIPELINE II=S
                
               
                
    //             data_t temp_products[S];
    //             #pragma HLS ARRAY_PARTITION variable=temp_products complete dim=1
                
    //             int temp_j[S];
    //             #pragma HLS ARRAY_PARTITION variable=temp_j complete dim=1
                
    //             //Manual Unrolling since loop bounds are unknown
    //             for (int s = 0; s < S; s++) {
    //                 #pragma HLS UNROLL
    //                 if (idx_B + s < local_col_ptr_B[k + 1]) {
    //                     temp_j[s] = local_row_indices_B[idx_B + s];
    //                     temp_products[s] = value_A * local_values_B[idx_B + s];
    //                 } else {
    //                     temp_j[s] = -1;
    //                     temp_products[s] = 0;
    //                 }
    //             }
                

    //             data_t current_values[S];
    //             #pragma HLS ARRAY_PARTITION variable=current_values complete dim=1
                
    //             for (int s = 0; s < S; s++) {
    //                 #pragma HLS UNROLL
    //                 if (temp_j[s] != -1) {
    //                     current_values[s] = local_C[i][temp_j[s]];
    //                 } else {
    //                     current_values[s] = 0;
    //                 }
    //             }
                

    //             for (int s = 0; s < S; s++) {
    //                 #pragma HLS UNROLL
    //                 if (temp_j[s] != -1) {
    //                     local_C[i][temp_j[s]] = current_values[s] + temp_products[s];
    //                 }
    //             }
    //         }
    //     }
    // }
   
    for (int i = 0; i < N; i++) {
        // Initialize row i of local_C.
        for (int j = 0; j < K; j++) {
            #pragma HLS unroll
            local_C[i][j] = 0;
        }
    
        // Iterate over nonzeros in row i of A.
        for (int idx_A = local_row_ptr_A[i]; idx_A < local_row_ptr_A[i+1]; idx_A++) {
          //  #pragma HLS LOOP_TRIPCOUNT min=1 max=64 avg=16
            int colA = local_column_indices_A[idx_A];
            data_t valA = local_values_A[idx_A];
    
            // Process the colA-th column of B in CSC format.
            for (int idx_B = local_col_ptr_B[colA]; idx_B < local_col_ptr_B[colA+1]; idx_B++) {
                #pragma HLS PIPELINE II=1
                int colB = local_row_indices_B[idx_B];
                // Make sure to index into local_C properly:
                local_C[i][colB] += valA * local_values_B[idx_B];
            }
        }


        for (int j = 0; j < K; j++) {
            #pragma HLS PIPELINE II=1
            C[i][j] = local_C[i][j];
        }
    }
     

    }
  