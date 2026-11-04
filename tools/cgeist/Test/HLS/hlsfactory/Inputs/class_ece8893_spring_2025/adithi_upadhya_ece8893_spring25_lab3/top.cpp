#include "dcl.h"

void load_matrix_A(const data_t values_A[N * M], const int column_indices_A[N * M], const int row_ptr_A[N + 1],
    data_t local_values_A[N * M], int local_column_indices_A[N * M], int local_row_ptr_A[N + 1]) {
    //load A (CSR format) and its components into local buffers
    for (int i = 0; i < N + 1; i++) {
        #pragma HLS UNROLL //full unroll
        local_row_ptr_A[i] = row_ptr_A[i];
    }
    for (int i = 0; i < row_ptr_A[N]; i++) {
        #pragma HLS UNROLL factor=16
        #pragma HLS performance target_tl=5000
        local_values_A[i] = values_A[i];
        local_column_indices_A[i] = column_indices_A[i];
    }

    
}

void load_matrix_B(const data_t values_B[M * K], const int row_indices_B[M * K], const int col_ptr_B[M + 1],
    data_t local_values_B[M * K], int local_row_indices_B[M * K], int local_col_ptr_B[M + 1]) {
    //load B (CSC format) and its components into local buffers
    for (int i = 0; i < M + 1; i++) {
        #pragma HLS UNROLL
        local_col_ptr_B[i] = col_ptr_B[i];
    }
    for (int i = 0; i < col_ptr_B[M]; i++) {
        #pragma HLS UNROLL factor=16 //match A's parallelism
        #pragma HLS performance target_tl=5000
        local_values_B[i] = values_B[i];
        local_row_indices_B[i] = row_indices_B[i];
    }
   
}

void multiply_matrices(const data_t local_values_A[N * M], const int local_column_indices_A[N * M], const int local_row_ptr_A[N + 1],
    const data_t local_values_B[M * K], const int local_row_indices_B[M * K], const int local_col_ptr_B[M + 1],
    data_t local_C[N][K]) {
    //sparse matrix multiplication: A (CSR) * B (CSC) = C (Dense)
    //#pragma HLS INLINE OFF //disable function inlining, preserve hierarchy

    data_t accumulators[N][K] = {0};
    #pragma HLS ARRAY_PARTITION variable=accumulators complete dim=2
    #pragma HLS BIND_STORAGE variable=accumulators type=ram_2p impl=bram

    Row: for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=3
        #pragma HLS LOOP_FLATTEN OFF
           
        ColA: for (int idx_A = local_row_ptr_A[i]; idx_A < local_row_ptr_A[i+1]; idx_A++) {
            #pragma HLS LOOP_TRIPCOUNT min=32 max=64
            const int k = local_column_indices_A[idx_A];
            const data_t a_val = local_values_A[idx_A];
            
            RowB: for (int idx_B = local_col_ptr_B[k]; idx_B < local_col_ptr_B[k+1]; idx_B++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS DEPENDENCE variable=accumulators inter WAR false
                const int j = local_row_indices_B[idx_B];
                accumulators[i][j] += a_val * local_values_B[idx_B];
            }
        }
    }

    WriteC: for (int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < K; j++) {
            local_C[i][j] = accumulators[i][j];
        }
    }
}

void store_result(const data_t local_C[N][K], data_t C[N][K]) {
    for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL factor=4
        for (int j = 0; j < K; j++) {
            #pragma HLS UNROLL factor=8
            C[i][j] = local_C[i][j];
        }
    }
}

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

    data_t local_values_A[N * M];
    int local_column_indices_A[N * M];
    int local_row_ptr_A[N + 1];

    data_t local_values_B[M * K];
    int local_row_indices_B[M * K];
    int local_col_ptr_B[M + 1];

    data_t local_C[N][K];

    #pragma HLS ARRAY_PARTITION variable=local_values_A cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=local_column_indices_A cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=local_row_ptr_A complete dim=1

    #pragma HLS ARRAY_PARTITION variable=local_values_B cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=local_row_indices_B cyclic factor=16 dim=1
    #pragma HLS ARRAY_PARTITION variable=local_col_ptr_B complete dim=1

    #pragma HLS ARRAY_PARTITION variable=local_C block factor=16 dim=2


    load_matrix_A(values_A, column_indices_A, row_ptr_A, local_values_A, local_column_indices_A, local_row_ptr_A);
    load_matrix_B(values_B, row_indices_B, col_ptr_B, local_values_B, local_row_indices_B, local_col_ptr_B);

    multiply_matrices(local_values_A, local_column_indices_A, local_row_ptr_A,
        local_values_B, local_row_indices_B, local_col_ptr_B,
        local_C);

    store_result(local_C, C); 
}