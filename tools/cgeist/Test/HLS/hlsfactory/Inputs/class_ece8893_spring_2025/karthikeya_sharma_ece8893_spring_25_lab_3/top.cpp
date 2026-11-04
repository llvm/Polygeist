// ECE8893 Parallel Programming for FPGA Lab-3 : Efficient Implementation of Sparse Matrix Multiplication
// Author: Karthikeya Sharma
// Description : This code contains the implementation for sparse_matrix_multiply_HLS kernel. To achieve a desirable performance, concepts such as Task Level Parallelism and Streaming have been implemented.
// Last Modified : 3/29/2025

#include "dcl.h"

void stream_matrix_A_rows (int row_ptr_A_BRAM[N+1], data_t values_A_BRAM[N*M], int col_indices_A_BRAM[N*M],
                           hls::stream<data_t, M> &values_A_stream, hls::stream<int, M> &column_indices_A_stream)
{
    for (int i = 0; i < N; i++)
    {   
        for (int j = row_ptr_A_BRAM[i]; j < row_ptr_A_BRAM[i + 1]; j++)
        {
            //Stream every row's contents for computation
            values_A_stream << values_A_BRAM[j];
            column_indices_A_stream << col_indices_A_BRAM[j];
        }
    }
}

void calculate_per_C_row (int row_ptr_A_BRAM[N + 1], 
                          data_t values_B_BRAM[M * K], int row_idx_B_BRAM[M * K], int col_ptr_B_BRAM[M + 1],
                          hls::stream<data_t, M> &values_A_stream, hls::stream<int, M> &column_indices_A_stream,
                          hls::stream<hls::vector<data_t, K>, 10> &output_C_row_stream)
{
    int col_idx_A;
    data_t value_A;

    for (int i = 0; i < N; i++)
    {
        hls::vector<data_t, K> output_C_row = (data_t)0;

        for (int j = row_ptr_A_BRAM[i]; j < row_ptr_A_BRAM[i + 1]; j++)
        {   
            column_indices_A_stream >> col_idx_A;
            values_A_stream >> value_A;

            for (int k = col_ptr_B_BRAM[col_idx_A]; k < col_ptr_B_BRAM[col_idx_A + 1]; k++)
            {
                int row_idx_B = row_idx_B_BRAM[k];
                data_t value_B = values_B_BRAM[k];

                output_C_row[row_idx_B] += value_A * value_B;
            }
        }

        //Stream the vector of accumulated indices in a row to the global memory
        output_C_row_stream << output_C_row;
    }
}

void accumulate_C_matrix(hls::stream<hls::vector<data_t, K>, 10> &output_C_row_stream, data_t C[N][K])
{
    for (int i = 0; i < N; i++)
    {
        hls::vector<data_t, K> C_row = (data_t) 0;
        output_C_row_stream >> C_row;

        for (int col = 0; col < K; ++col)
        {
            C[i][col] = C_row[col];
        }
    }
}

void sparse_matmul ( data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1], int row_ptr_A_copy[N + 1], 
                     data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1], data_t C[N][K])
{
#pragma HLS DATAFLOW
    hls::stream<data_t, M> values_A_row_stream;
    hls::stream<int, M> column_indices_A_row_stream;
    hls::stream<hls::vector<data_t, K>, 10> dense_row;

    //Stream every row
    stream_matrix_A_rows(row_ptr_A, values_A, column_indices_A, values_A_row_stream, column_indices_A_row_stream);

    //Calculate output_C[i, 0:N] -- dense matrix output for every row and stream it
    calculate_per_C_row(row_ptr_A_copy, values_B, row_indices_B, col_ptr_B, values_A_row_stream, column_indices_A_row_stream, dense_row);

    //Accumulate the rows of the streamed output_C[i, 0:N] matrix 
    accumulate_C_matrix(dense_row, C);
}

//Load Sparse Matrix A into BRAM
void load_Sparse_A(data_t values_A[N * M], data_t values_A_BRAM[N * M], 
                   int column_indices_A[N * M], int column_indices_A_BRAM[N * M],
                   int row_ptr_A[N + 1], int row_ptr_A_BRAM[N + 1])
{
    for (int i = 0; i < N*M; i++){
        values_A_BRAM[i] = values_A[i];
        column_indices_A_BRAM[i] = column_indices_A[i];
    }

    for (int i = 0; i < N+1; i++)
    {
        row_ptr_A_BRAM[i] = row_ptr_A[i];
    }
}

//Load Sparse Matrix B into BRAM
void load_Sparse_B(data_t values_B[M * K], data_t values_B_BRAM[M * K], 
                   int row_indices_B[M * K], int row_indices_B_BRAM[M * K],
                   int col_ptr_B[M + 1], int col_ptr_B_BRAM[M + 1])
{
    for (int i = 0; i < K*M; i++)
    {
        values_B_BRAM[i] = values_B[i];
        row_indices_B_BRAM[i] = row_indices_B[i];
    }

    for (int i = 0; i < M+1; i++)
    {
        col_ptr_B_BRAM[i] = col_ptr_B[i];
    }
}

// Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
void sparse_matrix_multiply_HLS(data_t values_A[N * M], 
                                int column_indices_A[N * M], 
                                int row_ptr_A[N + 1],
                                data_t values_B[M * K], 
                                int row_indices_B[M * K], 
                                int col_ptr_B[M + 1], 
                                data_t C[N][K]) 
{
//Matrix A
#pragma HLS interface m_axi port=values_A offset=slave bundle=mem1 max_widen_bitwidth=512 max_read_burst_length=256
#pragma HLS interface m_axi port=column_indices_A offset=slave bundle=mem1 max_widen_bitwidth=512 max_read_burst_length=256
#pragma HLS interface m_axi port=row_ptr_A offset=slave bundle=mem1 max_widen_bitwidth=512 max_read_burst_length=256
//Matrix B
#pragma HLS interface m_axi port=values_B offset=slave bundle=mem2 max_read_burst_length=256 max_widen_bitwidth=512
#pragma HLS interface m_axi port=row_indices_B offset=slave bundle=mem2 max_read_burst_length=256 max_widen_bitwidth=512
#pragma HLS interface m_axi port=col_ptr_B offset=slave bundle=mem2 max_read_burst_length=256 max_widen_bitwidth=512
//Matrix C
#pragma HLS interface m_axi port=C offset=slave bundle=mem3 max_write_burst_length=256 max_widen_bitwidth=512
#pragma HLS interface s_axilite port=return

    data_t values_A_BRAM[N*M];
    int column_indices_A_BRAM[N*M];
    int row_ptr_A_BRAM[N+1];

    data_t values_B_BRAM[N*M];
    int row_indices_B_BRAM[N*M];
    int col_ptr_B_BRAM[M+1];

    //Load sparse matrices onto BRAM
    load_Sparse_A(values_A, values_A_BRAM, 
                  column_indices_A, column_indices_A_BRAM,
                  row_ptr_A, row_ptr_A_BRAM);
                    
    load_Sparse_B(values_B, values_B_BRAM, 
                  row_indices_B, row_indices_B_BRAM,
                  col_ptr_B, col_ptr_B_BRAM);

    // Perform Sparse x Sparse Multiplication
    sparse_matmul(values_A_BRAM, column_indices_A_BRAM, row_ptr_A_BRAM, row_ptr_A_BRAM,
            values_B_BRAM, row_indices_B_BRAM, col_ptr_B_BRAM, C);
}