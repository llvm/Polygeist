/* NEG: row count is an expression but row i reads rowPtr[i + 1]..rowPtr[i + 2]
   (an expression-bound extent must not absorb an index shift). */
void neg_csr_rows_shifted(int m, const int *rowPtr, const int *colInd, const double *val,
                          const double *x, double *y) {
  for (int i = 0; i < m - 1; ++i) {
    double acc = 0.0;
    for (int k = rowPtr[i + 1]; k < rowPtr[i + 2]; ++k)
      acc += val[k] * x[colInd[k]];
    y[i] = acc;
  }
}
