/* NEG: also scatters into z, which no spec writes. */
void neg_csc_twotargets(int n, const int *colPtr, const int *rowInd, const double *val,
                        const double *x, double *y, double *z) {
  for (int j = 0; j < n; ++j)
    for (int k = colPtr[j]; k < colPtr[j + 1]; ++k) {
      y[rowInd[k]] += val[k] * x[j];
      z[rowInd[k]] += val[k];
    }
}
