//===========================================================================
// @file: spmm.h
// @brief: Header file for the sparse matrix multiplication function
//===========================================================================

#ifndef SPMM_H
#define SPMM_H

#include "dcl.h"
#include <hls_stream.h>


void read_csr_row
(
    int row_ptr_A_local[N + 1],
    data_t values_A_local[N * M],
    int col_idx_A_local[N * M],
    hls::stream<data_t, M> &values_A_row_stream,
    hls::stream<int, M> &column_indices_A_row_stream
);

void calc_dense_row
(
    int row_ptr_A_local[N + 1],
    data_t values_B_local[M * K],
    int row_idx_B_local[M * K],
    int col_ptr_B_local[M + 1],
    hls::stream<data_t, M> &values_A_row_stream,
    hls::stream<int, M> &column_indices_A_row_stream,
    hls::stream<hls::vector<data_t, K>, 3> &dense_row
);

void send_dense_row
(   
    hls::stream<hls::vector<data_t, K>, 3> &dense_row,
    data_t C[N][K]
);

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
);

#endif // SPMM_H