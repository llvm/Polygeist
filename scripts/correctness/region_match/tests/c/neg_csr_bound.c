/* NEG: upper bound rowPtr[i+2] is not a CSR row. */
void neg_csr_bound(int m, const int *rowPtr, const int *colInd, const double *val,
                   const double *x, double *y) {
  for (int i = 0; i < m; ++i) {
    double acc = 0.0;
    for (int k = rowPtr[i]; k < rowPtr[i + 2]; ++k) acc += val[k] * x[colInd[k]];
    y[i] = acc;
  }
}
