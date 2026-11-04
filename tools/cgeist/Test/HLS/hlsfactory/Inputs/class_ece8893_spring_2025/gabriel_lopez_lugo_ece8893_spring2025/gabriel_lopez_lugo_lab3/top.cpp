
#include "dcl.h"


void load_indices(packed_32_idx src[NUM_VECS_IDX], int dst[N * M]) {
    LOAD_VECS_INT:
    for (int i = 0; i < NUM_VECS_IDX; i++) {
#pragma HLS pipeline II=1
        packed_32_idx index_buf = src[i];        

        LOOP_VEC_INT:
        for (int j = 0; j < NUM_PACKED_IDX; j++) {
            dst[i * NUM_PACKED_IDX + j] = index_buf[j];
        }
    }
}

void load_values(packed_32_data_t src[NUM_VECS_DATA_T], data_t dst[N * M]) {
    LOAD_VECS_DATA_T:
    for (int i = 0; i < NUM_VECS_DATA_T; i++) {
#pragma HLS pipeline II=1
        packed_32_data_t data_buf = src[i];

        LOAD_VEC_DATA_T:
        for (int j = 0; j < NUM_PACKED_DATA_T; j++) {
            dst[i * NUM_PACKED_DATA_T + j] = data_buf[j];
        }
    }
}

void load_ptr(int src[N + 1], int dst[N + 1]) {
    LOAD_PTR:
    for (int i = 0; i < N + 1; i++) {
#pragma HLS pipeline II=1
        dst[i] = src[i];
    }
}

void store_values(data_t src[N * K], packed_32_data_t dst[NUM_VECS_DATA_T]) {
    STORE_VECS_DATA_T:
    for (int i = 0; i < NUM_VECS_DATA_T; i++) {
#pragma HLS pipeline II=1
        packed_32_data_t store_buf;

        STORE_VEC_DATA_T:
        for (int j = 0; j < NUM_PACKED_DATA_T; j++) {
            store_buf[j] = src[i * NUM_PACKED_DATA_T + j];
        }
        dst[i] = store_buf;
    }
}

void sparse_matrix_multiply_HLS(packed_32_data_t values_A[NUM_VECS_DATA_T], packed_32_idx column_indices_A[NUM_VECS_IDX], int row_ptr_A[N + 1], 
                                packed_32_data_t values_B[NUM_VECS_DATA_T], packed_32_idx row_indices_B[NUM_VECS_IDX], int col_ptr_B[M + 1], packed_32_data_t C[NUM_VECS_DATA_T])
{
#pragma HLS interface mode=axis port=values_A //offset=slave bundle=mem1
#pragma HLS interface mode=axis port=column_indices_A //offset=slave bundle=mem2
#pragma HLS interface m_axi port=row_ptr_A offset=slave bundle=mem3

#pragma HLS interface mode=axis port=values_B //offset=slave bundle=mem4
#pragma HLS interface mode=axis port=row_indices_B //offset=slave bundle=mem5
#pragma HLS interface m_axi port=col_ptr_B offset=slave bundle=mem6

#pragma HLS interface mode=axis port=C //offset=slave bundle=mem7

#pragma HLS interface s_axilite port=return

    // Perform sparse matrix multiplication
    data_t values_A_local[N * M];
    int row_ptr_A_local[N + 1];
    int column_indices_A_local[N * M];
#pragma HLS array_partition variable=values_A_local factor=8 cyclic
#pragma HLS array_partition variable=row_ptr_A_local factor=8 cyclic
#pragma HLS array_partition variable=column_indices_A_local factor=8 cyclic

    data_t values_B_local[M * K];
    int col_ptr_B_local[M + 1];
    int row_indices_B_local[M * K];
#pragma HLS array_partition variable=values_B_local factor=8 cyclic
#pragma HLS array_partition variable=col_ptr_B_local factor=8 cyclic
#pragma HLS array_partition variable=row_indices_B_local factor=8 cyclic
    
    // Statically defined to reduce LUT usage for initialization
    static data_t C_local[N * K] = {};
#pragma HLS array_partition variable=C_local factor=8 cyclic

    // Load matrix A data from CSR format
    load_values(values_A, values_A_local);
    load_indices(column_indices_A, column_indices_A_local);
    load_ptr(row_ptr_A, row_ptr_A_local);

    // Load matrix B data from CSC format
    load_values(values_B, values_B_local);
    load_indices(row_indices_B, row_indices_B_local);
    load_ptr(col_ptr_B, col_ptr_B_local);


    A_TIMES_B:
    for (int i = 0; i < N; i++) {
#pragma HLS unroll factor=8
                
        ITER_A:
        for (int idx_A = row_ptr_A_local[i]; idx_A < row_ptr_A_local[i + 1]; idx_A++) {
#pragma HLS unroll factor=8
            int k = column_indices_A_local[idx_A]; // Column index of A
            data_t value_A = values_A_local[idx_A];

            // Iterate over columns of B corresponding to row k
            ITER_B:
            for (int idx_B = col_ptr_B_local[k]; idx_B < col_ptr_B_local[k + 1]; idx_B++) {
#pragma HLS pipeline II=1
                int j = row_indices_B_local[idx_B]; // Row index of B
                data_t value_B = values_B_local[idx_B];

                // Accumulate the product into C[i][j]
                C_local[i * N + j] += value_A * value_B;
            }
        }
    }

    // Store the result in C
    store_values(C_local, C);
}