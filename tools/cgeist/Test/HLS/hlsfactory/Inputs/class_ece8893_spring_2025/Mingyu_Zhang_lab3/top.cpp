#include "dcl.h"

void copy_to_BRAM(data_t values_A[lim],       int column_indices_A[lim],       int row_ptr_A[N + 1],
                  data_t values_B[lim],       int row_indices_B[lim],          int col_ptr_B[M + 1],
                  data_t values_A_local[lim], int column_indices_A_local[lim], int row_ptr_A_local[N + 1],
                  data_t values_B_local[lim], int row_indices_B_local[lim],    int col_ptr_B_local[M + 1]){
    // Copy to BRAM
    copy_A:
    for (int i = 0; i < lim; i++){
    #pragma HLS pipeline II=1
        values_A_local[i] = values_A[i];
    }
    for (int i = 0; i < lim; i++){
    #pragma HLS pipeline II=1
        column_indices_A_local[i] = column_indices_A[i];
    }
    for (int i = 0; i < N+1; i++){
    #pragma HLS pipeline II=1
        row_ptr_A_local[i] = row_ptr_A[i];
    }
    
    copy_B:
    for (int i = 0; i < lim; i++){
    #pragma HLS pipeline II=1
        values_B_local[i] = values_B[i];
    }
    for (int i = 0; i < lim; i++){
    #pragma HLS pipeline II=1
        row_indices_B_local[i] = row_indices_B[i];
    }
    for (int i = 0; i < M+1; i++){
    #pragma HLS pipeline II=1
        col_ptr_B_local[i] = col_ptr_B[i];
    }
}

void compute_row(int i,
                 data_t values_A_local[lim], int column_indices_A_local[lim], int row_ptr_A_local[N+1],
                 data_t values_B_local[lim], int row_indices_B_local[lim],    int col_ptr_B_local[M+1],
                 data_t C_row[K]){
    for (int j = 0; j < K; j++){
    #pragma HLS pipeline II=1
        C_row[j] = 0;
    }
    
    for (int idx_A = row_ptr_A_local[i]; idx_A < row_ptr_A_local[i + 1]; idx_A++) {
    #pragma HLS pipeline II=1
        int k = column_indices_A_local[idx_A]; // Column index of A
        data_t value_A = values_A_local[idx_A];

        // Iterate over columns of B corresponding to row k
        for (int idx_B = col_ptr_B_local[k]; idx_B < col_ptr_B_local[k + 1]; idx_B++) {
        #pragma HLS pipeline II=1
        #pragma HLS unroll factor=8
            int j = row_indices_B_local[idx_B]; // Row index of B
            data_t value_B = values_B_local[idx_B];

            // Accumulate the product into C[i][j]
            C_row[j] += value_A * value_B;
        }
    }
}

void compute(data_t values_A_local[lim], int column_indices_A_local[lim], int row_ptr_A_local[N + 1],
             data_t values_B_local[lim], int row_indices_B_local[lim],    int col_ptr_B_local[M + 1],
             data_t C[N][K]){
    for (int i = 0; i < N; i++) {
    #pragma HLS dataflow
        data_t C_row[K];
        
    #pragma HLS array_partition variable=C_row dim=1 complete
        compute_row(i,
                    values_A_local, column_indices_A_local, row_ptr_A_local,
                    values_B_local, row_indices_B_local,    col_ptr_B_local,
                    C_row);
        for (int j = 0; j < K; j++){
        #pragma HLS pipeline II=1
            C[i][j] = C_row[j];
        }
    }
}

// Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
void sparse_matrix_multiply_HLS(data_t values_A[lim], int column_indices_A[lim], int row_ptr_A[N + 1],
                                data_t values_B[lim], int row_indices_B[lim],    int col_ptr_B[M + 1],
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
    
#pragma HLS dataflow
    
    data_t values_A_local[lim];
    int column_indices_A_local[lim];
    int row_ptr_A_local[N+1];
    
    data_t values_B_local[lim];
    int row_indices_B_local[lim];
    int col_ptr_B_local[M+1];
    
#pragma HLS array_partition variable=values_A_local         dim=1 factor=partf cyclic
#pragma HLS array_partition variable=column_indices_A_local dim=1 factor=partf cyclic
#pragma HLS array_partition variable=row_ptr_A_local        dim=1 complete
#pragma HLS array_partition variable=values_B_local         dim=1 factor=partf cyclic
#pragma HLS array_partition variable=row_indices_B_local    dim=1 factor=partf cyclic
#pragma HLS array_partition variable=col_ptr_B_local        dim=1 complete
    
    
    copy_to_BRAM(values_A, column_indices_A, row_ptr_A,
                 values_B, row_indices_B, col_ptr_B,
                 values_A_local, column_indices_A_local, row_ptr_A_local,
                 values_B_local, row_indices_B_local, col_ptr_B_local);
    
    // Perform Sparse x Sparse Multiplication
    compute(values_A_local, column_indices_A_local, row_ptr_A_local,
            values_B_local, row_indices_B_local,    col_ptr_B_local,
            C);

}

