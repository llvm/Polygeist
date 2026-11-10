/* cuSPARSE SpMV, CSR with CUSPARSE_INDEX_BASE_ONE, y = A*x: rowPtr and
   colInd hold 1-based positions (rowPtr[0] == 1). */
void sp_csr_onebased(int m, const int *rowPtr, const int *colInd, const double *val,
                     const double *x, double *y) {
  for (int i = 0; i < m; ++i) {
    double acc = 0.0;
    for (int k = rowPtr[i] - 1; k < rowPtr[i + 1] - 1; ++k)
      acc += val[k] * x[colInd[k] - 1];
    y[i] = acc;
  }
}
