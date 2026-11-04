
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>

typedef ap_fixed<16, 5> data_t;

#define N 64  // Rows of A and C
#define M 64  // Columns of A and Rows of B
#define K 64  // Columns of B and C
#define SPARSITY 0.1  // Percentage of non-zero elements
#define lim 435 // SPARSITY=0.1: 435; SPARSITY=0.5: 2065; SPARSITY=0.8: 3315
#define partf 16


void sparse_matrix_multiply_HLS(data_t values_A[lim], int column_indices_A[lim], int row_ptr_A[N + 1],
                             data_t values_B[lim], int row_indices_B[lim], int col_ptr_B[M + 1], data_t C[N][K]);

