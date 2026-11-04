#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>
#include <hls_stream.h>

typedef ap_fixed<16, 5> data_t;
typedef ap_uint<512> mem_t;

#define N 64  // Rows of A and C
#define M 64  // Columns of A and Rows of B
#define K 64  // Columns of B and C
#define SPARSITY 0.5  // Percentage of non-zero elements

#define MPDA ((N * M) / 32)
#define MPPA ((N + 1) / 32) + 1
#define MPDB ((M * K) / 32)
#define MPPB ((M + 1) / 32) + 1
#define DPM 32

void sparse_matrix_multiply_HLS(mem_t values_A[MPDA], mem_t column_indices_A[MPDA], ap_uint<16> row_ptr_A[N + 1], 
                                mem_t values_B[MPDB], mem_t row_indices_B[MPDB], ap_uint<16> col_ptr_B[M + 1],
                                mem_t C[N][K/DPM]);


