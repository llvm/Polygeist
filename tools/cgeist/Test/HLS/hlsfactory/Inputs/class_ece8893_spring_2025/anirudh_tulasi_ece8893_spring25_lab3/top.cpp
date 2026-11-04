#include "dcl.h"

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

    data_t values_A_local[N * M], values_B_local[M * K];
    int column_indices_A_local[N * M], row_ptr_A_local[N + 1], row_indices_B_local[M * K], col_ptr_B_local[M + 1];
    
    // Array partitioning for input arrays
    #pragma HLS array_partition variable=values_A_local cyclic factor=4
    #pragma HLS array_partition variable=column_indices_A_local cyclic factor=4
    #pragma HLS array_partition variable=values_B_local cyclic factor=4
    #pragma HLS array_partition variable=row_indices_B_local cyclic factor=4
    #pragma HLS array_partition variable=row_ptr_A_local complete
    #pragma HLS array_partition variable=col_ptr_B_local complete

    for (int i = 0; i < N * M; i++) {
        #pragma HLS pipeline II=1 style=frp
        #pragma HLS unroll factor=2
        values_A_local[i] = values_A[i];
        column_indices_A_local[i] = column_indices_A[i];
    }
    for (int i = 0; i < M * K; i++) {
        #pragma HLS pipeline II=1 style=frp
        values_B_local[i] = values_B[i];
        row_indices_B_local[i] = row_indices_B[i];
    }
    for (int i = 0; i < N + 1; i++) {
        #pragma HLS pipeline II=1 style=frp
        row_ptr_A_local[i] = row_ptr_A[i];
    }
    for (int i = 0; i < M + 1; i++) {
        #pragma HLS pipeline II=1 style=frp
        col_ptr_B_local[i] = col_ptr_B[i];
    }

    for (int i = 0; i < N; i++) {
        data_t C_local[K];
        #pragma HLS array_partition variable=C_local complete

        for (int j = 0; j < K; j++) {
            #pragma HLS unroll
            C_local[j] = 0;
        }

        for (int idx_A = row_ptr_A_local[i]; idx_A < row_ptr_A_local[i + 1]; idx_A++) {
            int k = column_indices_A_local[idx_A];
            data_t value_A = values_A_local[idx_A];

            for (int idx_B = col_ptr_B_local[k]; idx_B < col_ptr_B_local[k + 1]; idx_B++) {
                #pragma HLS pipeline II=1 style=frp
                #pragma HLS unroll factor=2  
                int j = row_indices_B_local[idx_B];
                data_t value_B = values_B_local[idx_B];
                data_t product = value_A * value_B;
                C_local[j] += product;           
            }
        }

        for (int j = 0; j < K; j++) {
            #pragma HLS pipeline II=1 style=frp
            C[i][j] = C_local[j];
        }
    }
}
