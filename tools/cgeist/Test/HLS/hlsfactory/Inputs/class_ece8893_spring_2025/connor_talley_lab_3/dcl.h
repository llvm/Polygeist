
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

// typedef ap_fixed<16, 5> data_t;
#define DATA_WIDTH 16
#define DATA_INT 5
typedef ap_fixed<DATA_WIDTH, DATA_INT> data_t;

#define INDEX_WIDTH 32
typedef int coord_t;

#define N 64  // Rows of A and C
#define M 64  // Columns of A and Rows of B
#define K 64  // Columns of B and C
#define AXI_WIDTH 512
//may want separate block sizes for data and index due to their different widths, but that might be a later optimization
#define LOAD_BLOCK_SIZE (AXI_WIDTH / DATA_WIDTH)
// #define N 5  // Rows of A and C
// #define M 5  // Columns of A and Rows of B
// #define K 5  // Columns of B and C
#define SPARSITY 0.80  // Percentage of non-zero elements
typedef struct {
    hls::vector<data_t, 2> values;
    hls::vector<coord_t, 2> destination;
} mult_op_t;


void sparse_matrix_multiply_HLS(data_t values_A[N * M], int column_indices_A[N * M], hls::vector<int, N+1> row_ptr_A, 
                             data_t values_B[M * K], int row_indices_B[M * K], hls::vector<int, M+1> col_ptr_B, hls::vector<data_t, K> C[N]);


