/* cuSPARSE SpMV, CSR with CUSPARSE_INDEX_64I, y = A*x. */
#include <stdint.h>
void sp_csr_i64(int64_t m, const int64_t *rowPtr, const int64_t *colInd, const double *val,
                const double *x, double *y) {
  for (int64_t i = 0; i < m; ++i) {
    double acc = 0.0;
    for (int64_t k = rowPtr[i]; k < rowPtr[i + 1]; ++k)
      acc += val[k] * x[colInd[k]];
    y[i] = acc;
  }
}
