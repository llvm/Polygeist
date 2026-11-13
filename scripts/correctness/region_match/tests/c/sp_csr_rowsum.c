/* cuSPARSE SpMV, CSR, y = A*x: y[i] = sum_{k=rowPtr[i]}^{rowPtr[i+1]-1} val[k]*x[colInd[k]] */
void sp_csr_rowsum(int m, const int *rowPtr, const int *colInd, const double *val,
                   const double *x, double *y) {
  for (int i = 0; i < m; ++i) {
    double acc = 0.0;
    for (int k = rowPtr[i]; k < rowPtr[i + 1]; ++k)
      acc += val[k] * x[colInd[k]];
    y[i] = acc;
  }
}
