void batched_gemv_as_gemm(int batches, int rows, int cols,
                          double A[rows][cols],
                          double X[batches][cols],
                          double Y[batches][rows]) {
  for (int b = 0; b < batches; ++b) {
    for (int i = 0; i < rows; ++i) {
      Y[b][i] = 0.0;
    }
  }

  for (int b = 0; b < batches; ++b) {
    for (int i = 0; i < rows; ++i) {
      for (int k = 0; k < cols; ++k) {
        Y[b][i] += X[b][k] * A[i][k];
      }
    }
  }
}
