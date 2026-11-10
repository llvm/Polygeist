/* y = A*x for CSC: clear y, then scatter. */
void sp_csc_zeroed(int m, int n, const int *colPtr, const int *rowInd, const double *val,
                   const double *x, double *restrict y) {
  for (int i = 0; i < m; ++i) y[i] = 0.0;
  for (int j = 0; j < n; ++j)
    for (int k = colPtr[j]; k < colPtr[j + 1]; ++k)
      y[rowInd[k]] += val[k] * x[j];
}
