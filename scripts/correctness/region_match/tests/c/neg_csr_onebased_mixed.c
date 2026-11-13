/* NEG: 1-based row pointers but 0-based column indices (neither CSR convention). */
void neg_csr_onebased_mixed(int m, const int *rowPtr, const int *colInd, const double *val,
                            const double *x, double *y) {
  for (int i = 0; i < m; ++i) {
    double acc = 0.0;
    for (int k = rowPtr[i] - 1; k < rowPtr[i + 1] - 1; ++k)
      acc += val[k] * x[colInd[k]];
    y[i] = acc;
  }
}
