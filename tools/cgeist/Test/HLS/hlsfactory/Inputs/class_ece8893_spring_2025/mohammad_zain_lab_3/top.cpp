#include "dcl.h"
#define UNROLL_FACTOR 2
// Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
void sparse_matrix_multiply_HLS(hls::vector<data_t_reduced, N> values_A[M], hls::vector<int16_t, N> column_indices_A[M], int16_t row_ptr_A[N+1], 
                             hls::vector<data_t_reduced, K> values_B[M], hls::vector<int16_t, K> row_indices_B[M], int16_t col_ptr_B[M + 1], hls::vector<data_t, K> C[N]) 
{
#pragma HLS interface m_axi port=values_A offset=slave bundle=mem1 channel=0
#pragma HLS interface m_axi port=column_indices_A offset=slave bundle=mem1 channel=1
#pragma HLS interface m_axi port=row_ptr_A offset=slave bundle=mem4 channel=0

#pragma HLS interface m_axi port=values_B offset=slave bundle=mem2 channel=0
#pragma HLS interface m_axi port=row_indices_B offset=slave bundle=mem2 channel=1
#pragma HLS interface m_axi port=col_ptr_B offset=slave bundle=mem2 channel=2

#pragma HLS interface m_axi port=C offset=slave bundle=mem3

#pragma HLS interface s_axilite port=return



int16_t row_ptr_A_local[N+1];
#pragma HLS array_partition variable=row_ptr_A_local cyclic factor=UNROLL_FACTOR
hls::vector<data_t_reduced, N> values_A_local[M];
//#pragma HLS array_partition variable=values_A_local cyclic factor=UNROLL_FACTOR
hls::vector<int16_t, N> column_indices_A_local[M];
//#pragma HLS array_partition variable=column_indices_A_local cyclic factor=UNROLL_FACTOR

load_row_ptr_A:
for(int i=0;i<N+1;i++){
    row_ptr_A_local[i] = row_ptr_A[i];
}

//only load neceesary data
int index_A = row_ptr_A_local[N]%N?row_ptr_A_local[N]/N+1:row_ptr_A_local[N]/N;
load_column_indices_A:
for(int i=0;i<index_A;i++){
    column_indices_A_local[i] = column_indices_A[i];
    values_A_local[i] = values_A[i];
}

hls::vector<data_t_reduced, K> values_B_local[M];
//#pragma HLS array_partition variable=values_B_local cyclic factor=UNROLL_FACTOR
hls::vector<int16_t, K> row_indices_B_local[M];
//#pragma HLS array_partition variable=row_indices_B_local cyclic factor=UNROLL_FACTOR
int16_t col_ptr_B_local[M+1];
#pragma HLS array_partition variable=row_ptr_A_local cyclic factor=UNROLL_FACTOR

load_row_indices_B:
for(int i=0;i<M+1;i++){
    col_ptr_B_local[i] = col_ptr_B[i];
}

//only load neceesary data
int index_B = col_ptr_B_local[M]%M?col_ptr_B_local[M]/M+1:col_ptr_B_local[M]/M;
load_values_B:
for(int i=0;i<index_B;i++){
    row_indices_B_local[i] = row_indices_B[i];
    values_B_local[i] = values_B[i];
}

hls::vector<data_t, K> C_local[N];

    compute_main_loop:
    for (int i = 0; i < N; i++) {
       #pragma HLS UNROLL factor=UNROLL_FACTOR skip_exit_check
        compute_inner_loop1:
        for (int idx_A = row_ptr_A_local[i]; idx_A < row_ptr_A_local[i + 1]; idx_A++) {
           // #pragma HLS loop_tripcount min=0 max=64 //sparsity of 10, 50 and 80%
            int index_A=idx_A/N;
            int index_B=idx_A%N;
            int k = column_indices_A_local[index_A][index_B]; // Column index of A
           compute_inner_loop2:
          //  change to definite loop bound
            for (int idx_B = col_ptr_B_local[k]; idx_B < col_ptr_B_local[k + 1]; idx_B++) {
               // #pragma HLS loop_tripcount min=0 max=64 //sparsity of 10, 50 and 80%
                int index_A_2=idx_B/M;
                int index_B_2=idx_B%M;
                int j = row_indices_B_local[index_A_2][index_B_2]; // Row index of B
                data_t_reduced product = values_A_local[index_A][index_B] * values_B_local[index_A_2][index_B_2];
                C_local[i][j] += product;
            }
            // compute_inner_loop2:
            // for (int idx_B = 0; idx_B < M; idx_B++) {
            //     #pragma HLS UNROLL factor=32 skip_exit_check
            //     if(idx_B >=col_ptr_B_local[k] && idx_B <col_ptr_B_local[k + 1])
            //     {
            //     int index_A_2=idx_B/M;
            //     int index_B_2=idx_B%M;
            //     int j = row_indices_B_local[index_A_2][index_B_2]; // Row index of B
            //     data_t_reduced product = values_A_local[index_A][index_B] * values_B_local[index_A_2][index_B_2];
            //     C_local[i][j] += product;
            //     }
            // }
        }
    }

    //copy to DRAM
    copy_results_to_DRAM:
    for (int i = 0; i < N; i++) {
            C[i] = C_local[i];//write to vector
    }
    
}
