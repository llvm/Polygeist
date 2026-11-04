#include "dcl.h"

#define NUM_DUPLICATES 8 // Number of duplicate data copies for parallel processing

void load_data_mem1(const data_t* values_A, const int* column_indices_A, const int* row_ptr_A,
                    data_t values_A_local[NUM_DUPLICATES][N * M], int column_indices_A_local[NUM_DUPLICATES][N * M], int row_ptr_A_local[NUM_DUPLICATES][N + 1]) {
#pragma HLS array_partition variable=row_ptr_A_local complete dim=1
#pragma HLS array_partition variable=values_A_local complete dim=1
#pragma HLS array_partition variable=column_indices_A_local complete dim=1

    
    // Load row pointers for the first duplicate
    for (int i = 0; i < N + 1; i++) {
        row_ptr_A_local[0][i] = row_ptr_A[i];
    }
    // Load values and column indices for the first duplicate
    for (int i = 0; i < row_ptr_A_local[0][N]; i++) {
        values_A_local[0][i] = values_A[i];
        column_indices_A_local[0][i] = column_indices_A[i];
    }
    // Duplicate data across NUM_DUPLICATES for parallel processing
    for (int d = 1; d < NUM_DUPLICATES; d++) {
        for (int i = 0; i < N + 1; i++) {
            row_ptr_A_local[d][i] = row_ptr_A_local[0][i];
        }
        for (int i = 0; i < row_ptr_A_local[0][N]; i++) {
            values_A_local[d][i] = values_A_local[0][i];
            column_indices_A_local[d][i] = column_indices_A_local[0][i];
        }
    }
}

void load_data_mem2(const data_t* values_B, const int* row_indices_B, const int* col_ptr_B,
                    data_t values_B_local[NUM_DUPLICATES][M * K], int row_indices_B_local[NUM_DUPLICATES][M * K], int col_ptr_B_local[NUM_DUPLICATES][M + 1]) {
#pragma HLS array_partition variable=col_ptr_B_local complete dim=1
#pragma HLS array_partition variable=values_B_local complete dim=1
#pragma HLS array_partition variable=row_indices_B_local complete dim=1

    // Load column pointers for the first duplicate
    for (int i = 0; i < M + 1; i++) {
        col_ptr_B_local[0][i] = col_ptr_B[i];
    }
    // Load values and row indices for the first duplicate
    for (int i = 0; i < col_ptr_B_local[0][M]; i++) {
        values_B_local[0][i] = values_B[i];
        row_indices_B_local[0][i] = row_indices_B[i];
    }
    // Duplicate data across NUM_DUPLICATES for parallel processing
    for (int d = 1; d < NUM_DUPLICATES; d++) {
        for (int i = 0; i < M + 1; i++) {
            col_ptr_B_local[d][i] = col_ptr_B_local[0][i];
        }
        for (int i = 0; i < col_ptr_B_local[0][M]; i++) {
            values_B_local[d][i] = values_B_local[0][i];
            row_indices_B_local[d][i] = row_indices_B_local[0][i];
        }
    }
}

void sparse_matrix_multiply_core(int d, 
                                 int row_ptr_A_local[N + 1], 
                                 int col_ptr_B_local[M + 1],
                                 data_t values_A_local[N * M],
                                 int column_indices_A_local[N * M],
                                 data_t values_B_local[M * K],
                                 int row_indices_B_local[M * K],
                                 data_t C_local[N][K]) {
    // Loop over the rows assigned to this duplicate
    for (int i = d * (N / NUM_DUPLICATES); i < (d + 1) * (N / NUM_DUPLICATES); i++) {
        // Process each nonzero element in row i of matrix A
        for (int idx_A = row_ptr_A_local[i]; idx_A < row_ptr_A_local[i + 1]; idx_A++) {
            int k = column_indices_A_local[idx_A]; // Column index of the element in A
            data_t value_A = values_A_local[idx_A]; // Value of the element in A

            // Multiply with corresponding nonzero elements in column k of matrix B
            for (int idx_B = col_ptr_B_local[k]; idx_B < col_ptr_B_local[k + 1]; idx_B++) {
#pragma HLS pipeline 
                int j = row_indices_B_local[idx_B];  // Row index of the element in B
                data_t value_B = values_B_local[idx_B]; // Value of the element in B
                C_local[i][j] += value_A * value_B; // Accumulate the product in result matrix C
            }
        }
    }
}

void merge_results(data_t C[N][K], data_t C_local[NUM_DUPLICATES][N][K]) {
    // Iterate over all elements of the result matrix
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            int d = i / (N / NUM_DUPLICATES); // Determine the duplicate responsible for this row
            C[i][j] = C_local[d][i][j]; // Assign the computed value from the corresponding duplicate
        }
    }
}

// Top-level function for sparse matrix multiplication
void sparse_matrix_multiply_HLS(data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1],
                                data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1],
                                data_t C[N][K]) {

#pragma HLS interface m_axi port=values_A offset=slave bundle=mem1
#pragma HLS interface m_axi port=column_indices_A offset=slave bundle=mem1
#pragma HLS interface m_axi port=row_ptr_A offset=slave bundle=mem1

#pragma HLS interface m_axi port=values_B offset=slave bundle=mem2
#pragma HLS interface m_axi port=row_indices_B offset=slave bundle=mem2
#pragma HLS interface m_axi port=col_ptr_B offset=slave bundle=mem2

#pragma HLS interface m_axi port=C offset=slave bundle=mem3

#pragma HLS interface s_axilite port=return

#pragma HLS dataflow // Enable dataflow optimization

    // Local buffers for matrix A, B, and C
    // These buffers are duplicated for parallel processing
    int row_ptr_A_local[NUM_DUPLICATES][N + 1];
    int col_ptr_B_local[NUM_DUPLICATES][M + 1];
    data_t values_A_local[NUM_DUPLICATES][N * M];
    int column_indices_A_local[NUM_DUPLICATES][N * M];
    data_t values_B_local[NUM_DUPLICATES][M * K];
    int row_indices_B_local[NUM_DUPLICATES][M * K];
    data_t C_local[NUM_DUPLICATES][N][K];

#pragma HLS array_partition variable=C_local complete dim=1

    // Load data from global memory to local memory
    load_data_mem1(values_A, column_indices_A, row_ptr_A, values_A_local, column_indices_A_local, row_ptr_A_local);
    load_data_mem2(values_B, row_indices_B, col_ptr_B, values_B_local, row_indices_B_local, col_ptr_B_local);

    // Perform sparse matrix multiplication in parallel for all duplicates
    for (int d = 0; d < NUM_DUPLICATES; d++) {
#pragma HLS unroll  // Unroll the loop for parallel execution
        sparse_matrix_multiply_core(d, row_ptr_A_local[d], col_ptr_B_local[d], values_A_local[d],
                                    column_indices_A_local[d], values_B_local[d], row_indices_B_local[d],
                                    C_local[d]);
    }

    // Merge partial results into the final result matrix
    merge_results(C, C_local);
}