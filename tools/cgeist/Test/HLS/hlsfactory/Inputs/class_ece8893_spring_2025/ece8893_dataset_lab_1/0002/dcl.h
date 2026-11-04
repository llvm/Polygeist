
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>

typedef ap_fixed<16, 5> data_t;
typedef ap_uint<512> wide_t; // 512-bit wide access (512 / 16 = 32 data_t values per transfer)

#define N 64  // Rows of A and C
#define M 64  // Columns of A and Rows of B
#define K 64  // Columns of B and C
#define SPARSITY 0.80  // Percentage of non-zero elements

void sparse_matrix_multiply_HLS(
    wide_t values_A[N * M / 32], int column_indices_A[N * M], int row_ptr_A[N + 1],
    wide_t values_B[M * K / 32], int row_indices_B[M * K], int col_ptr_B[M + 1],
    data_t C[N][K]);

