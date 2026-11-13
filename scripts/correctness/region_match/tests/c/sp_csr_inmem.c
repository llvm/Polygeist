/* y += A*x for CSR, accumulating directly in memory (no scalar temporary). */
void sp_csr_inmem(int m, const int *rowPtr, const int *colInd, const double *val,
                  const double *x, double *restrict y) {
  for (int i = 0; i < m; ++i)
    for (int k = rowPtr[i]; k < rowPtr[i + 1]; ++k)
      y[i] += x[colInd[k]] * val[k];
}
