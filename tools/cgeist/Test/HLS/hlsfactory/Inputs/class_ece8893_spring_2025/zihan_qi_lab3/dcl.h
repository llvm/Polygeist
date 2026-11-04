
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <hls_vector.h>
#include <hls_stream.h>
#include <stdlib.h>
#include <cstdint>

typedef ap_fixed<16, 5> data_t;
typedef hls::vector<data_t, 32> vec32_t;  // 32 fixed_t = 512-bit
typedef hls::vector<int, 16> vec16_int_t;  

#define N 64  // Rows of A and C
#define M 64  // Columns of A and Rows of B
#define K 64  // Columns of B and C
#define SPARSITY 0.1  // Percentage of non-zero elements
//#define len_10 4096

#define non_zero 432  // S=0.1
//#define non_zero 2058 // S=0.5
//#define non_zero 3313 // S=0.8
#define f 16

// void sparse_matrix_multiply_HLS(vec32_t values_A[len_10 / 32], vec16_int_t column_indices_A[len_10 / 16], vec16_int_t row_ptr_A[N / 16], 
//                              vec32_t values_B[len_10 / 32], vec16_int_t row_indices_B[len_10 / 16], vec16_int_t col_ptr_B[M / 16], data_t C[N][K]);

// void sparse_matrix_multiply_HLS(vec32_t values_A[len_10 / 32], vec16_int_t column_indices_A[len_10 / 16], vec16_int_t row_ptr_A[N / 16], 
//                              vec32_t values_B[len_10 / 32], vec16_int_t row_indices_B[len_10 / 16], vec16_int_t col_ptr_B[M / 16], vec32_t C[N][K / 32]);
// void sparse_matrix_multiply_HLS(data_t values_A[N * M], int column_indices_A[N * M], vec16_int_t row_ptr_A[N / 16], 
//                              data_t values_B[M * K], int row_indices_B[M * K], vec16_int_t col_ptr_B[M / 16], vec32_t C[N][K / 32]);
void sparse_matrix_multiply_HLS(
    data_t values_A[non_zero], int column_indices_A[non_zero], vec16_int_t row_ptr_A[N / 16], 
    data_t values_B[non_zero], int row_indices_B[non_zero], vec16_int_t col_ptr_B[M / 16], vec32_t C[N][K / 32]);