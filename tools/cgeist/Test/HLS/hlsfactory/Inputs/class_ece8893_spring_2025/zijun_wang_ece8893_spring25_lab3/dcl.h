
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>

typedef ap_fixed<16, 5> data_t;
typedef ap_int<32> data_int;

#define N 64  // Rows of A and C
#define M 64  // Columns of A and Rows of B
#define K 64  // Columns of B and C
#define SPARSITY 0.1  // Percentage of non-zero elements
#define num_A_words (N*M)/32
#define num_A_int_words (N*M)/16
#define num_A_ptr_words N/16
#define num_B_words (M*K)/32
#define num_B_int_words (M*K)/16
#define num_B_ptr_words M/16
#define num_C_row_words K/32
#define P 2
// #define MAX_NNZ_PER_ROW 64
// #define MAX_NNZ_PER_COL 64
// #define TILE_SIZE 16

void sparse_matrix_multiply_HLS(data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1], 
                             data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1], data_t C[N][K]);


