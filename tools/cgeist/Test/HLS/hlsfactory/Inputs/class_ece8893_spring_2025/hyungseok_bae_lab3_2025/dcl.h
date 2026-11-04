#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>
#include <hls_stream.h>
#include <hls_vector.h>

#define VECTOR_SIZE 8
#define HASH_SIZE 128 // 해시 테이블 크기 (조정 가능)

typedef ap_fixed<16, 5> data_t;

#define N 64  // Rows of A and C
#define M 64  // Columns of A and Rows of B
#define K 64  // Columns of B and C
#define SPARSITY 0.1  // Percentage of non-zero elements

// Vectorized data types for wide transactions
typedef hls::vector<data_t, VECTOR_SIZE> vec_data_t;
typedef hls::vector<int, VECTOR_SIZE> vec_idx_t;

void sparse_matrix_multiply_HLS(
    const vec_data_t values_A[], const vec_idx_t column_indices_A[], const int row_ptr_A[],
    const vec_data_t values_B[], const vec_idx_t row_indices_B[], const int col_ptr_B[],
    hls::vector<data_t, K> C_vec[N]);

