/* cuSPARSE SpMV, CSC, y += A*x: column j scatters val[k]*x[j] into y[rowInd[k]]. */
void sp_csc_scatter(int n, const int *colPtr, const int *rowInd, const double *val,
                    const double *x, double *restrict y) {
  for (int j = 0; j < n; ++j) {
    double xj = x[j];
    for (int k = colPtr[j]; k < colPtr[j + 1]; ++k)
      y[rowInd[k]] = y[rowInd[k]] + xj * val[k];
  }
}
