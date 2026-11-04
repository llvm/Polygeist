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

data_t values_A_l[N * M];
int column_indices_A_l[N * M];
data_t values_B_l[M * K];
int row_indices_B_l[M * K];
int row_ptr_A_l[N + 1];
int col_ptr_B_l[M + 1];
data_t C_l[N][K];

    // Copy to Local BRAM
    for(int i = 0; i < N*M; i++){
        values_A_l[i] = values_A[i];
        column_indices_A_l[i] = column_indices_A[i];
    }
    
    for(int i = 0; i < M*K; i++){
        values_B_l[i] = values_B[i];
        row_indices_B_l[i] = row_indices_B[i];
    }
   

    for(int i = 0; i < N + 1; i++){
        row_ptr_A_l[i] = row_ptr_A[i];
    }

    for(int i = 0; i < M + 1; i++){
        col_ptr_B_l[i] = col_ptr_B[i];
    }
 

    // Perform Sparse x Sparse Multiplication
    for (int i = 0; i < N; i++) {
        for (int idx_A = row_ptr_A_l[i]; idx_A < row_ptr_A_l[i + 1]; idx_A++) {
            int k = column_indices_A_l[idx_A]; // Column index of A
            data_t value_A = values_A_l[idx_A];

            // Iterate over columns of B corresponding to row k
            for (int idx_B = col_ptr_B_l[k]; idx_B < col_ptr_B_l[k + 1]; idx_B++) {

                int j = row_indices_B_l[idx_B]; // Column index of B
                data_t value_B = values_B_l[idx_B];

                // Accumulate the product into C[i][j]
                // std::cout << i << value_A << " " << value_B << std::endl;
                C_l[i][j] += value_A * value_B;
            }
        }
    }

    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < K; j++)
        C[i][j] = C_l[i][j];
    }

}
