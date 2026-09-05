#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef POLYGEIST_SOLVER_N
#define POLYGEIST_SOLVER_N 40
#endif

extern void kernel_cholesky(int, double (*)[POLYGEIST_SOLVER_N]);

static void initialize(int n, double a[n][n]) {
  double(*base)[n] = malloc(sizeof(double[n][n]));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j)
      base[i][j] = (double)(-j % n) / n + 1.0;
    for (int j = i + 1; j < n; ++j)
      base[i][j] = 0.0;
    base[i][i] = 1.0;
  }
  for (int r = 0; r < n; ++r)
    for (int s = 0; s < n; ++s) {
      a[r][s] = 0.0;
      for (int t = 0; t < n; ++t)
        a[r][s] += base[r][t] * base[s][t];
    }
  free(base);
}

static void reference(int n, double a[n][n]) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      for (int k = 0; k < j; ++k)
        a[i][j] -= a[i][k] * a[j][k];
      a[i][j] /= a[j][j];
    }
    for (int k = 0; k < i; ++k)
      a[i][i] -= a[i][k] * a[i][k];
    a[i][i] = sqrt(a[i][i]);
  }
}

int main(void) {
  const int n = POLYGEIST_SOLVER_N;
  double(*got)[n] = malloc(sizeof(double[n][n]));
  double(*want)[n] = malloc(sizeof(double[n][n]));
  initialize(n, got);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      want[i][j] = got[i][j];
  kernel_cholesky(n, got);
  reference(n, want);
  double max_abs = 0.0, max_rel = 0.0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j <= i; ++j) {
      double error = fabs(got[i][j] - want[i][j]);
      double relative = error / fmax(1.0, fabs(want[i][j]));
      max_abs = fmax(max_abs, error);
      max_rel = fmax(max_rel, relative);
    }
  int pass = isfinite(max_abs) && max_rel <= 1.0e-10;
  printf("POLYBENCH_CHOLESKY_%s n=%d max_abs=%.17g max_rel=%.17g\n",
         pass ? "PASS" : "FAIL", n, max_abs, max_rel);
  free(got);
  free(want);
  return pass ? 0 : 1;
}
