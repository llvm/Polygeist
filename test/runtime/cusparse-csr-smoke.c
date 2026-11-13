#include "polygeist_cublas_rt.h"

#include <math.h>
#include <stdio.h>

int main(void) {
  const int rowptr[] = {0, 2, 4, 5};
  const int cols[] = {0, 2, 1, 2, 0};
  const double values[] = {2.0, 1.0, 3.0, -1.0, 4.0};
  const double x[] = {5.0, 7.0, 11.0};
  double y[] = {0.0, 0.0, 0.0};
  polygeist_cublas_init();
  /* y = 1*A*x + 0*y (y is not read). */
  polygeist_cusparse_spmv_csr_f64_sized(
      3, 4, rowptr, 5, cols, 5, values, 3, x, 3, y, 1.0, 0.0);
  const double expected[] = {21.0, 10.0, 20.0};
  for (int i = 0; i < 3; ++i) {
    if (fabs(y[i] - expected[i]) > 1.0e-12) {
      fprintf(stderr, "row %d: got %.17g expected %.17g\n",
              i, y[i], expected[i]);
      return 1;
    }
  }
  /* y = 2*A*x + 1*y: alpha/beta are honoured, not fixed to 1/0. */
  polygeist_cusparse_spmv_csr_f64_sized(
      3, 4, rowptr, 5, cols, 5, values, 3, x, 3, y, 2.0, 1.0);
  polygeist_cublas_destroy();
  const double accumulated[] = {63.0, 30.0, 60.0};
  for (int i = 0; i < 3; ++i) {
    if (fabs(y[i] - accumulated[i]) > 1.0e-12) {
      fprintf(stderr, "row %d (alpha=2, beta=1): got %.17g expected %.17g\n",
              i, y[i], accumulated[i]);
      return 1;
    }
  }
  puts("cuSPARSE CSR SpMV: PASS");
  return 0;
}
