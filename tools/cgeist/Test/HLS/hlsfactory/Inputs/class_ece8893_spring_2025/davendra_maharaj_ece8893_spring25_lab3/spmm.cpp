//===========================================================================
// @file: spmm.cpp
// @brief: Sparse matrix multiplication function implementation
//===========================================================================

#include "spmm.h"


void spmm
(
    data_t values_A[N * M], 
    int column_indices_A[N * M], 
    int row_ptr_A[N + 1],
    int row_ptr_A_copy[N + 1],
    data_t values_B[M * K], 
    int row_indices_B[M * K], 
    int col_ptr_B[M + 1], 
    data_t C[N][K]
)
{
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<data_t, M> values_A_row_stream;
    hls::stream<int, M> column_indices_A_row_stream;
    hls::stream<hls::vector<data_t, K>, 3> dense_row;

    // Read the row of A
    read_csr_row(row_ptr_A, values_A, column_indices_A, values_A_row_stream, column_indices_A_row_stream);

    // Calculate the dense row
    calc_dense_row(row_ptr_A_copy, values_B, row_indices_B, col_ptr_B, values_A_row_stream, column_indices_A_row_stream, dense_row);

    // Send the dense row to the output
    send_dense_row(dense_row, C);
}

void read_csr_row
(
    int row_ptr_A_local[N + 1],
    data_t values_A_local[N * M],
    int col_idx_A_local[N * M],
    hls::stream<data_t, M> &values_A_row_stream,
    hls::stream<int, M> &column_indices_A_row_stream
)
{
#pragma HLS INLINE off

    ROW: for ( int row = 0; row < N; ++ row )
    {   
        // Iterate over each non-zero element in the current row of A
        ROW_PTR_A: for ( int i = row_ptr_A_local[row]; i < row_ptr_A_local[row + 1]; ++i )
        {
            // Write the value of the current non-zero element in the current row of A to the values_A_stream
            values_A_row_stream << values_A_local[i];

            // Write the column index of the current non-zero element in the current row of A to the column_indices_A_stream
            column_indices_A_row_stream << col_idx_A_local[i];
        }
    }
}

void calc_dense_row
(
    int row_ptr_A_local[N + 1],
    data_t values_B_local[M * K],
    int row_idx_B_local[M * K],
    int col_ptr_B_local[M + 1],
    hls::stream<data_t, M> &values_A_row_stream,
    hls::stream<int, M> &column_indices_A_row_stream,
    hls::stream<hls::vector<data_t, K>, 3> &dense_row
)
{
#pragma HLS INLINE off

    ROW: for ( int row = 0; row < N; ++row )
    {   
        hls::vector<data_t, K> dense_row_vec = (data_t) 0;

        ROW_PTR_A: for ( int i = row_ptr_A_local[row]; i < row_ptr_A_local[row + 1]; ++i )
        {   
            int col_idx_A = column_indices_A_row_stream.read();
            data_t value_A = values_A_row_stream.read();

            COL_PTR_B: for ( int col = col_ptr_B_local[col_idx_A]; col < col_ptr_B_local[col_idx_A + 1]; ++col )
            {
                int row_idx_B = row_idx_B_local[col];
                data_t value_B = values_B_local[col];

                dense_row_vec[row_idx_B] += value_A * value_B;
            }
        }
        dense_row << dense_row_vec;
    }
}

void send_dense_row
(   
    hls::stream<hls::vector<data_t, K>, 3> &dense_row,
    data_t C[N][K]
)
{
#pragma HLS INLINE off

    // Iterate over each row of C
    ROW: for ( int row = 0; row < N; ++ row )
    {
        hls::vector<data_t, K> C_row = (data_t) 0;

        dense_row >> C_row;

        // Write the row of C to the output
        WRITE_C: for ( int col = 0; col < K; ++ col )
        {
            C[row][col] = C_row[col];
        }
    }
}