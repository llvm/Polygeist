/* CSR y = A*x with the row count passed by reference. */
void sp_csr_sizeptr(const int *mp, const int *rowPtr, const int *colInd, const double *val,
                    const double *x, double *y) {
  int m = *mp;
  for (int i = 0; i < m; ++i) {
    double acc = 0.0;
    for (int k = rowPtr[i]; k < rowPtr[i + 1]; ++k)
      acc += val[k] * x[colInd[k]];
    y[i] = acc;
  }
}
