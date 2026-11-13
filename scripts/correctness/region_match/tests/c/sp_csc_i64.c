/* cuSPARSE SpMV, CSC with CUSPARSE_INDEX_64I, y = A*x + y (beta = 1). */
void sp_csc_i64(long n, const long *colPtr, const long *rowInd, const double *val,
                const double *x, double *y) {
  for (long j = 0; j < n; ++j)
    for (long k = colPtr[j]; k < colPtr[j + 1]; ++k)
      y[rowInd[k]] += val[k] * x[j];
}
