/* NEG: x[k] instead of x[colInd[k]] (no gather). */
void neg_csr_nogather(int m, const int *rowPtr, const int *colInd, const double *val,
                      const double *x, double *y) {
  for (int i = 0; i < m; ++i) {
    double acc = 0.0;
    for (int k = rowPtr[i]; k < rowPtr[i + 1]; ++k) acc += val[k] * x[k];
    y[i] = acc;
  }
}
