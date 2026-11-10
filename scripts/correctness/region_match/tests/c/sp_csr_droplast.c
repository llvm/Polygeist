/* CSR y = A*x over the first m - 1 rows only (row count is an expression). */
void sp_csr_droplast(int m, const int *rowPtr, const int *colInd, const double *val,
                     const double *x, double *y) {
  for (int i = 0; i < m - 1; ++i) {
    double acc = 0.0;
    for (int k = rowPtr[i]; k < rowPtr[i + 1]; ++k)
      acc += val[k] * x[colInd[k]];
    y[i] = acc;
  }
}
