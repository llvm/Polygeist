#include "dcl_dense.h"


// Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
void sparse_matrix_multiply_HLS(data_t A[N][M],data_t B[M][K], data_t C[N][K]) 
{
    #pragma HLS interface m_axi port=A offset=slave bundle=mem1
    #pragma HLS interface m_axi port=B offset=slave bundle=mem2
    #pragma HLS interface m_axi port=C offset=slave bundle=mem3

    #pragma HLS interface s_axilite port=return
    data_t A_local[N][M];
    data_t B_local[M][N];
    data_t C_local[N][K];
    
    #pragma HLS array_partition variable=A_local dim=3 factor=8 cyclic
    #pragma HLS array_partition variable=B_local dim=2 factor=8 cyclic
    #pragma HLS array_partition variable=C_local dim=3 factor=8 cyclic

    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < M; j++)
        {
            #pragma HLS unroll
            A_local[i][j] = A[i][j];
            B_local[i][j] = B[i][j];
        }
    }


    // Perform Sparse x Sparse Multiplication
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < M; j++)
        {
            #pragma HLS PIPELINE
            for (int k = 0; k < K; k++)
            {
                #pragma HLS unroll factor=8
                C_local[i][j] += A_local[i][k] * B_local[k][j];
            }
        }
    }

    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < K; j++)
        {
            #pragma HLS unroll
            C[i][j] = C_local[i][j];
        }
    }
}
