#include "dcl.h"

// only represents 0-63
typedef ap_uint<6> axis_index_t;

// Sparse Matrix Multiplication: A (CSR) * B (CSC) = C (Dense)
void sparse_matrix_multiply_HLS(data_t values_A[N * M],
                                int column_indices_A[N * M],
                                int row_ptr_A[N + 1], data_t values_B[M * K],
                                int row_indices_B[M * K], int col_ptr_B[M + 1],
                                data_t C[N][K]) {
#pragma HLS interface m_axi port = values_A offset = slave bundle = mem1
#pragma HLS interface m_axi port = column_indices_A offset = slave bundle = mem2
#pragma HLS interface m_axi port = row_ptr_A offset = slave bundle = mem2

#pragma HLS interface m_axi port = values_B offset = slave bundle = mem3
#pragma HLS interface m_axi port = row_indices_B offset = slave bundle = mem4
#pragma HLS interface m_axi port = col_ptr_B offset = slave bundle = mem4

#pragma HLS interface m_axi port = C offset = slave bundle = mem5

#pragma HLS interface s_axilite port = return

  data_t values_A_loc[N * M];
  int column_indices_A_loc[N * M];
  int row_ptr_A_loc[N + 1];

  data_t values_B_loc[M * K];
  int row_indices_B_loc[M * K];
  int col_ptr_B_loc[M + 1];

  // this static keyword saves us from having to zero out C_loc
  // but it would be a problem if we ran this function multiple times
  // static data_t C_loc[N][K];
  static data_t C_row[K];

  static data_t B_T[K][M];
  static data_t A_row[M];

#pragma HLS ARRAY_PARTITION variable = B_T type = complete dim = 2
#pragma HLS ARRAY_PARTITION variable = A_row type = complete dim = 1

  // does turning these into for loops make them any faster for code analyzer?
  memcpy(values_B_loc, values_B, M * K * sizeof(data_t));
  memcpy(row_indices_B_loc, row_indices_B, M * K * sizeof(int));
  memcpy(col_ptr_B_loc, col_ptr_B, (M + 1) * sizeof(int));

  memcpy(values_A_loc, values_A, N * M * sizeof(data_t));
  memcpy(column_indices_A_loc, column_indices_A, N * M * sizeof(int));
  memcpy(row_ptr_A_loc, row_ptr_A, (N + 1) * sizeof(int));

  // Densify Matrix B
  // 7 bits is the minimum needed for values in the range 0 to 64

  // since B is transposed, we refer to the columns of B_T as the rows of B

BCol:
  for (int row = 0; row < K; ++row) {
    //#pragma HLS PIPELINE off
    // IDEA - this access pattern could be more linear?
    // and parallelized?
  BSlice:
    for (int i = col_ptr_B_loc[row]; i < col_ptr_B_loc[row + 1]; ++i) {
      //#pragma HLS PIPELINE off
      int col = row_indices_B_loc[i];
      data_t value = values_B_loc[i];
      // B[row][col] = value;
      B_T[col][row] = value;
    }
  }

  // Slice A Row by Row
  // Then A_Row x Dense_B_Mat

ARowS:
  for (int A_row_idx = 0; A_row_idx < N; ++A_row_idx) {
    // Can't pipeline
    //#pragma HLS PIPELINE off
    // IDEA - this access pattern could be more linear?
    // and parallelized?
  ARowZero:
    for (int m = 0; m < M; ++m) {
#pragma HLS UNROLL
      A_row[m] = 0;
    }

  ASlice:
    for (int i = row_ptr_A_loc[A_row_idx]; i < row_ptr_A_loc[A_row_idx + 1];
         ++i) {
      //#pragma HLS PIPELINE off
      int col = column_indices_A_loc[i];
      data_t value = values_A_loc[i];
      A_row[col] = value;
    }

  DotBCol:
    for (int B_col_idx = 0; B_col_idx < K; ++B_col_idx) {
      // Letting this auto pipeline gives crazy performance benefits
      data_t sum = 0;

    DotElem:
      for (int m = 0; m < M; ++m) {
#pragma HLS UNROLL
        sum += A_row[m] * B_T[B_col_idx][m];
      }
      // C_loc[A_row_idx][B_col_idx] = sum;
      C_row[B_col_idx] = sum;
    }

    memcpy(C[A_row_idx], C_row, K * sizeof(data_t));
  }

  // memcpy(C, C_loc, N * K * sizeof(data_t));
}