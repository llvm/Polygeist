#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef POLYGEIST_SOLVER_N
#define POLYGEIST_SOLVER_N 40
#endif

extern void kernel_trisolv(int, double (*)[POLYGEIST_SOLVER_N], double *,
                           double *);

static void initialize(int n, double l[n][n], double x[n], double b[n]) {
  for (int i = 0; i < n; ++i) {
    x[i] = -999.0;
    b[i] = (double)i;
    for (int j = 0; j <= i; ++j)
      l[i][j] = (double)(i + n - j + 1) * 2.0 / n;
  }
}

static void reference(int n, double l[n][n], double x[n], double b[n]) {
  for (int i = 0; i < n; ++i) {
    x[i] = b[i];
    for (int j = 0; j < i; ++j)
      x[i] -= l[i][j] * x[j];
    x[i] /= l[i][i];
  }
}

int main(void) {
  const int n = POLYGEIST_SOLVER_N;
  double(*l)[n] = calloc((size_t)n * n, sizeof(double));
  double *got = malloc(sizeof(double[n]));
  double *want = malloc(sizeof(double[n]));
  double *b = malloc(sizeof(double[n]));
  initialize(n, l, got, b);
  reference(n, l, want, b);
  kernel_trisolv(n, l, got, b);
  double max_abs = 0.0, max_rel = 0.0;
  for (int i = 0; i < n; ++i) {
    double error = fabs(got[i] - want[i]);
    double relative = error / fmax(1.0, fabs(want[i]));
    max_abs = fmax(max_abs, error);
    max_rel = fmax(max_rel, relative);
  }
  int pass = isfinite(max_abs) && max_rel <= 1.0e-10;
  printf("POLYBENCH_TRISOLV_%s n=%d max_abs=%.17g max_rel=%.17g\n",
         pass ? "PASS" : "FAIL", n, max_abs, max_rel);
  free(l);
  free(got);
  free(want);
  free(b);
  return pass ? 0 : 1;
}
