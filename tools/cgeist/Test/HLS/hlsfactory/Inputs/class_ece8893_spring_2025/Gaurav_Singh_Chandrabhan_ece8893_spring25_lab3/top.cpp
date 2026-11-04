#include "dcl.h"

void load_A_mat_sparse(data_t values_A[N * M], data_t A_val_BRAM[N * M], int column_indices_A[N * M], int A_col_idx_BRAM[N * M],
                   int row_ptr_A[N + 1], int A_rptr_BRAM[N + 1])
{
    // Copy non-zero values and column indices from sparse matrix A to BRAM
    COPY_A_VALUES: for (int i = 0; i < N*M; i++){
        A_val_BRAM[i] = values_A[i];
        A_col_idx_BRAM[i] = column_indices_A[i];
    }

    // Copy row pointers from sparse matrix A to BRAM
    COPY_A_ROW_PTR: for (int i = 0; i < N+1; i++)
    {
        A_rptr_BRAM[i] = row_ptr_A[i];
    }
}

void load_B_mat_sparse(data_t values_B[M * K], data_t B_val_BRAM[M * K], int row_indices_B[M * K], int B_row_idx_BRAM[M * K],
                   int col_ptr_B[M + 1], int B_cptr_BRAM[M + 1])
{   
    // Copy non-zero values and row indices from sparse matrix B to BRAM
    COPY_B_VALUES: for (int i = 0; i < K*M; i++)
    {
        B_val_BRAM[i] = values_B[i];
        B_row_idx_BRAM[i] = row_indices_B[i];
    }

    // Copy column pointers from sparse matrix B to BRAM
    COPY_B_COL_PTR: for (int i = 0; i < M+1; i++)
    {
        B_cptr_BRAM[i] = col_ptr_B[i];
    }
}

void read_csr_row( int row_ptr_A_local[N + 1], data_t values_A_local[N * M], int col_idx_A_local[N * M],
    hls::stream<data_t, M> &A_val_row_stream, hls::stream<int, M> &A_col_idx_row_stream)
{
#pragma HLS INLINE off
    // Iterate over each row in matrix A
    READ_A_ROWS: for ( int row = 0; row < N; ++ row )
    {   
        // Iterate over each non-zero element in the current row of A
        READ_A_NON_ZERO: for ( int i = row_ptr_A_local[row]; i < row_ptr_A_local[row + 1]; ++i )
        {
            // Send the non-zero value and corresponding column index to streams
            A_val_row_stream << values_A_local[i];
            A_col_idx_row_stream << col_idx_A_local[i];
        }
    }
}

void calc_dense_row(int row_ptr_A_local[N + 1], data_t values_B_local[M * K], int row_idx_B_local[M * K], int col_ptr_B_local[M + 1], 
    hls::stream<data_t, M> &A_val_row_stream, hls::stream<int, M> &A_col_idx_row_stream,
    hls::stream<hls::vector<data_t, K>, 3> &dense_row
)
{
    // Iterate over each row of matrix A
    COMPUTE_DENSE_ROWS: for ( int row = 0; row < N; ++row)
    {   
        hls::vector<data_t, K> dense_row_vec = (data_t) 0;
        // Iterate over each non-zero element in row A
        READ_A_NON_ZERO: for ( int i = row_ptr_A_local[row]; i < row_ptr_A_local[row + 1]; ++i )
        {   
            int col_idx_A = A_col_idx_row_stream.read();
            data_t value_A = A_val_row_stream.read();

            // Iterate over non-zero elements in the corresponding column of B
            READ_B_NON_ZERO: for ( int col = col_ptr_B_local[col_idx_A]; col < col_ptr_B_local[col_idx_A + 1]; ++col )
            {
                int row_idx_B = row_idx_B_local[col];
                data_t value_B = values_B_local[col];
                // Perform multiplication and accumulate result in dense row
                dense_row_vec[row_idx_B] += value_A * value_B;
            }
        }

        // Send the computed dense row to the stream
        dense_row << dense_row_vec;
    }
}

void store_dense_row( hls::stream<hls::vector<data_t, K>, 3> &dense_row, data_t C[N][K] )
{
    // Iterate over each row of output matrix C
    WRITE_C_ROWS: for ( int row = 0; row < N; ++ row )
    {
        hls::vector<data_t, K> C_row = (data_t) 0;
        dense_row >> C_row;

        // Write the computed dense row into matrix C
        WRITE_C_COLS: for ( int col = 0; col < K; ++ col )
        {
            C[row][col] = C_row[col];
        }
    }
}

void helper_compute( data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1], int row_ptr_A_copy[N + 1],
    data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1], data_t C[N][K]
)
{
    // Enable parallel execution of reading, calculation and storing
    #pragma HLS DATAFLOW

    hls::stream<data_t, M> A_val_row_stream;
    hls::stream<int, M> A_col_idx_row_stream;
    hls::stream<hls::vector<data_t, K>, 3> dense_row;

    // Read the row of A
    read_csr_row(row_ptr_A, values_A, column_indices_A, A_val_row_stream, A_col_idx_row_stream);

    // Calculate the dense row
    calc_dense_row(row_ptr_A_copy, values_B, row_indices_B, col_ptr_B, A_val_row_stream, A_col_idx_row_stream, dense_row);

    // Store the dense row
    store_dense_row(dense_row, C);
}

// The main Sparse Matrix Multiplication Function: A (CSR) * B (CSC) = C (Dense)
void sparse_matrix_multiply_HLS(data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1],
                                data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1], 
                                data_t C[N][K]) 
{

    //Adjusted the max bitwidth and burst length for Matrix A
    #pragma HLS interface m_axi port=values_A offset=slave bundle=mem1 max_widen_bitwidth=512 max_read_burst_length=256
    #pragma HLS interface m_axi port=column_indices_A offset=slave bundle=mem1 max_widen_bitwidth=512 max_read_burst_length=256
    #pragma HLS interface m_axi port=row_ptr_A offset=slave bundle=mem1 max_widen_bitwidth=512 max_read_burst_length=256
    
    //Adjusted the max bitwidth and burst length for Matrix B
    #pragma HLS interface m_axi port=values_B offset=slave bundle=mem2 max_widen_bitwidth=512 max_read_burst_length=256
    #pragma HLS interface m_axi port=row_indices_B offset=slave bundle=mem2 max_widen_bitwidth=512 max_read_burst_length=256
    #pragma HLS interface m_axi port=col_ptr_B offset=slave bundle=mem2 max_widen_bitwidth=512 max_read_burst_length=256
    
    //Adjusted the max bitwidth and burst length for Matrix C
    #pragma HLS interface m_axi port=C offset=slave bundle=mem3 max_widen_bitwidth=512 max_write_burst_length=256

    #pragma HLS interface s_axilite port=return

    // BRAM Buffers for Faster Memory Access
    data_t A_val_BRAM[N*M];
    int A_col_idx_BRAM[N*M];
    int A_rptr_BRAM[N+1];

    data_t B_val_BRAM[N*M];
    int B_row_idx_BRAM[N*M];
    int B_cptr_BRAM[M+1];

    // Enable parallel execution of loading and computation
    #pragma HLS DATAFLOW

    // Load the matrices A and B on BRAM
    load_A_mat_sparse(values_A, A_val_BRAM, column_indices_A, A_col_idx_BRAM, row_ptr_A, A_rptr_BRAM);  
    load_B_mat_sparse(values_B, B_val_BRAM, row_indices_B, B_row_idx_BRAM, col_ptr_B, B_cptr_BRAM);

    // Using helper compute function to maintain the dataflow
    helper_compute(A_val_BRAM, A_col_idx_BRAM, A_rptr_BRAM, A_rptr_BRAM, B_val_BRAM, B_row_idx_BRAM, B_cptr_BRAM, C);
}