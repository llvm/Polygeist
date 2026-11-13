/* No restrict: cgeist re-reads colPtr[j+1] in the loop condition because y
   may alias it (scf.while); the counted-while support executes it as a for
   loop.  The match relies on the pipeline-wide noalias assumption (y does not
   overlap colPtr/rowInd/val/x), which the evidence states. */
void lim_csc_noalias(int n, const int *colPtr, const int *rowInd, const double *val,
                     const double *x, double *y) {
  for (int j = 0; j < n; ++j)
    for (int k = colPtr[j]; k < colPtr[j + 1]; ++k)
      y[rowInd[k]] += val[k] * x[j];
}
