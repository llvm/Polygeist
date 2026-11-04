
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

#define N 64  // Rows of A and C
#define M 64  // Columns of A and Rows of B
#define K 64  // Columns of B and C
#define SPARSITY 0.1  // Percentage of non-zero elements

#define UFACTOR 2 // unroll factor

#define LOGVECFACTOR 5
#define VECFACTOR (1<<LOGVECFACTOR) // Tile Factor
#define LOGR (12-LOGVECFACTOR) // R = 14 - log2(VECFACTOR)
#define R (1<<LOGR) // Restoverbits


// void sparse_matrix_multiply_HLS(data_t values_A[N * M], int column_indices_A[N * M], int row_ptr_A[N + 1], 
//                              data_t values_B[M * K], int row_indices_B[M * K], int col_ptr_B[M + 1], data_t C[N][K]);

// void sparse_matrix_multiply_HLS(hls::vector<data_t, N> values_A[M], hls::vector<int, N> column_indices_A[M], hls::vector<int, N + 1> row_ptr_A, 
                            //  hls::vector<data_t, M> values_B[K], hls::vector<int, M> row_indices_B[K], hls::vector<int, M + 1> col_ptr_B, data_t C[N][K]); 
// void sparse_matrix_multiply_HLS(hls::vector<data_t, N> values_A[M], hls::vector<int, N> column_indices_A[M], hls::vector<int, N + 1> row_ptr_A, 
                            //  hls::vector<data_t, M> values_B[K], hls::vector<int, M> row_indices_B[K], hls::vector<int, M + 1> col_ptr_B, hls::vector<data_t, K> C[N]); 
void sparse_matrix_multiply_HLS(hls::vector<data_t, VECFACTOR> values_A[R], hls::vector<int, VECFACTOR> column_indices_A[R], int row_ptr_A[N + 1], 
                             hls::vector<data_t, VECFACTOR> values_B[R], hls::vector<int, VECFACTOR> row_indices_B[R], int col_ptr_B[M + 1], hls::vector<data_t, K> C[N]); 

