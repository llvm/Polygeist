#include <cstddef>
#include "dcl.h"
#include "spmm.h"

// Load CSR matrix A
void loadCSR
(
    int nnz_A,
    data_t values_A[N * M], 
    int column_indices_A[N * M], 
    int row_ptr_A[N + 1], 
    data_t values_A_local[N * M], 
    int col_idx_A_local[N * M], 
    int row_ptr_A_local[N + 1],
    int row_ptr_A_local_copy[N + 1]
)
{
    for (int i = 0; i < nnz_A ; i++)
    {
        values_A_local[i] = values_A[i];
        col_idx_A_local[i] = column_indices_A[i];
    }

    for (int i = 0; i < N + 1; i++)
    {
        row_ptr_A_local[i] = row_ptr_A[i];
        row_ptr_A_local_copy[i] = row_ptr_A[i];
    }
}

// Load CSC matrix B
void loadCSC
(
    int nnz_B,
    data_t values_B[M * K], 
    int row_indices_B[M * K], 
    int col_ptr_B[M + 1], 
    data_t values_B_local[M * K], 
    int row_idx_B_local[M * K], 
    int col_ptr_B_local[M + 1]
)
{
    for (int i = 0; i < nnz_B; i++)
    {
        values_B_local[i] = values_B[i];
        row_idx_B_local[i] = row_indices_B[i];
    }

    for (int i = 0; i < M + 1; i++)
    {
        col_ptr_B_local[i] = col_ptr_B[i];
    }
}

// Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
void sparse_matrix_multiply_HLS
(
    int nnz_A,
    int nnz_B,
    data_t values_A[N * M], 
    int column_indices_A[N * M], 
    int row_ptr_A[N + 1], 
    data_t values_B[M * K], 
    int row_indices_B[M * K], 
    int col_ptr_B[M + 1],
    // hls::vector<data_t,K> C[N]
    data_t C[N][K]
)
{
#pragma HLS interface m_axi port=values_A offset=slave bundle=mem1 max_read_burst_length=256 max_widen_bitwidth=512
#pragma HLS interface m_axi port=column_indices_A offset=slave bundle=mem1 max_read_burst_length=256 max_widen_bitwidth=512
#pragma HLS interface m_axi port=row_ptr_A offset=slave bundle=mem1 max_read_burst_length=256 max_widen_bitwidth=512

#pragma HLS interface m_axi port=values_B offset=slave bundle=mem2 max_read_burst_length=256 max_widen_bitwidth=512
#pragma HLS interface m_axi port=row_indices_B offset=slave bundle=mem2 max_read_burst_length=256 max_widen_bitwidth=512
#pragma HLS interface m_axi port=col_ptr_B offset=slave bundle=mem2 max_read_burst_length=256 max_widen_bitwidth=512

#pragma HLS interface m_axi port=C offset=slave bundle=mem3 max_write_burst_length=256 max_widen_bitwidth=512

#pragma HLS interface s_axilite port=return

    // Local buffers for CSR A
    data_t values_A_local[N * M];
    int col_idx_A_local[N * M];
    int row_ptr_A_local[N + 1];
    int row_ptr_A_local_copy[N + 1];

    // Local buffers for CSC B
    data_t values_B_local[M * K];
    int row_idx_B_local[M * K];
    int col_ptr_B_local[M + 1];


    // Load CSR matrix A
    loadCSR(nnz_A, values_A, column_indices_A, row_ptr_A, values_A_local, col_idx_A_local, row_ptr_A_local, row_ptr_A_local_copy);

    // Load CSC matrix B
    loadCSC(nnz_B, values_B, row_indices_B, col_ptr_B, values_B_local, row_idx_B_local, col_ptr_B_local);

    // Perform Sparse Matrix Multiplication
    spmm
    (
        values_A_local, 
        col_idx_A_local, 
        row_ptr_A_local,
        row_ptr_A_local_copy, 
        values_B_local, 
        row_idx_B_local, 
        col_ptr_B_local, 
        C
    );
}
