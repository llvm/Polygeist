#include "../../runtime/polygeist_cublas_rt.h"

#include <math.h>
#include <stdio.h>

int main(void) {
  double x[4] = {1.0, 2.0, 3.0, 4.0};
  double y[4] = {5.0, 6.0, 7.0, 8.0};
  const double expected[4] = {11.5, 15.0, 18.5, 22.0};
  polygeist_cublas_daxpby(4, 1.5, x, 2.0, y);
  for (int i = 0; i < 4; ++i)
    if (fabs(y[i] - expected[i]) > 1.0e-12) {
      fprintf(stderr, "FAIL daxpy[%d]: got %.17g expected %.17g\n",
              i, y[i], expected[i]);
      return 1;
    }
  puts("PASS cublasDaxpby");
  return 0;
}
