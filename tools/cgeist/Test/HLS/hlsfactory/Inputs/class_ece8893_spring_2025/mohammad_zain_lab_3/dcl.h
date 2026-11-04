
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ap_fixed.h>
#include <hls_math.h>
#include <stdlib.h>
#include <cstdint>
#include <hls_vector.h>
#include <hls_stream.h>


typedef ap_fixed<16, 5> data_t;
typedef ap_fixed<14, 3> data_t_reduced;


#define N 64  // Rows of A and C
#define M 64  // Columns of A and Rows of B
#define K 64  // Columns of B and C
#define SPARSITY 0.1  // Percentage of non-zero elements


void sparse_matrix_multiply_HLS(hls::vector<data_t_reduced, N> values_A[M], hls::vector<int16_t, N> column_indices_A[M], int16_t row_ptr_A[N+1], 
    hls::vector<data_t_reduced, K> values_B[K], hls::vector<int16_t, K> row_indices_B[M], int16_t col_ptr_B[M + 1], hls::vector<data_t, K> C[N]);

