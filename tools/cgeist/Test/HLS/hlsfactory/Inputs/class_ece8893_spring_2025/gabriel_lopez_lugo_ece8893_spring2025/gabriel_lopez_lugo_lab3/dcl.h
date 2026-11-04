#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdlib.h>
#include <cstdint>

#include <ap_fixed.h>
#include <hls_math.h>
#include <hls_vector.h>

typedef ap_fixed<16, 5> data_t;

constexpr int N = 64;            // Rows of A and C
constexpr int M = 64;            // Columns of A and Rows of B
constexpr int K = 64;            // Columns of B and C
constexpr float SPARSITY = 0.5;  // Percentage of non-zero elements

constexpr int NUM_PACKED_IDX = 512 / (sizeof(short) * 8);   // Number of packed indices
constexpr int NUM_PACKED_DATA_T = 512 / data_t::width;      // Number of packed data_t

constexpr int MATRIX_SIZE = N * K; 

constexpr int NUM_VECS_IDX = MATRIX_SIZE / NUM_PACKED_IDX;          // Number of indices vectors
constexpr int NUM_VECS_DATA_T = MATRIX_SIZE / NUM_PACKED_DATA_T;    // Number of data_t vectors

using packed_32_idx = hls::vector<short, NUM_PACKED_IDX>;
using packed_32_data_t = hls::vector<data_t, NUM_PACKED_DATA_T>;

void sparse_matrix_multiply_HLS(packed_32_data_t values_A[NUM_VECS_DATA_T], packed_32_idx column_indices_A[NUM_VECS_IDX], int row_ptr_A[N + 1], 
                             packed_32_data_t values_B[NUM_VECS_DATA_T], packed_32_idx row_indices_B[NUM_VECS_IDX], int col_ptr_B[M + 1], packed_32_data_t C[NUM_VECS_DATA_T]);