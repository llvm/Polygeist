#include "polygeist_cublas_rt.h"

#include <math.h>
#include <stdio.h>

int main(void) {
  const double x[4] = {1.0, 2.0, 3.0, 4.0};
  const double y[4] = {5.0, 6.0, 7.0, 8.0};
  double result = 0.0;
  polygeist_cublas_dot_f64(4, x, y, &result);
  if (fabs(result - 70.0) > 1.0e-12) {
    fprintf(stderr, "FAIL cublasDdot: got %.17g, expected 70\n", result);
    return 1;
  }
  puts("PASS cublasDdot");
  return 0;
}
