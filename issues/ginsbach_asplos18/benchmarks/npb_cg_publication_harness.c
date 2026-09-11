#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void npb_cg_conj_grad_core(int n, int nnz, int *colidx, int *rowstr,
                           double *x, double *z, double *a, double *p,
                           double *q, double *r, double *rnorm);
void npb_cg_reference(int n, int nnz, int *colidx, int *rowstr, double *x,
                      double *z, double *a, double *p, double *q, double *r,
                      double *rnorm);

enum { kWarmups = 5, kSamples = 5, kN = 65536, kNnz = 3 * kN - 2 };

static double wall_ms(void) {
  struct timespec timestamp;
  clock_gettime(CLOCK_MONOTONIC, &timestamp);
  return timestamp.tv_sec * 1000.0 + timestamp.tv_nsec / 1.0e6;
}

int main(void) {
  int *colidx = malloc((size_t)kNnz * sizeof(int));
  int *rowstr = malloc(((size_t)kN + 1) * sizeof(int));
  double *a = malloc((size_t)kNnz * sizeof(double));
  double *x = malloc((size_t)kN * sizeof(double));
  double *z = malloc((size_t)kN * sizeof(double));
  double *p = malloc((size_t)kN * sizeof(double));
  double *q = malloc((size_t)kN * sizeof(double));
  double *r = malloc((size_t)kN * sizeof(double));
  double *expected_z = malloc((size_t)kN * sizeof(double));
  double *expected_p = malloc((size_t)kN * sizeof(double));
  double *expected_q = malloc((size_t)kN * sizeof(double));
  double *expected_r = malloc((size_t)kN * sizeof(double));
  if (!colidx || !rowstr || !a || !x || !z || !p || !q || !r ||
      !expected_z || !expected_p || !expected_q || !expected_r)
    return 100;

  int cursor = 0;
  rowstr[0] = 0;
  for (int row = 0; row < kN; ++row) {
    if (row != 0) {
      colidx[cursor] = row - 1;
      a[cursor++] = -0.25;
    }
    colidx[cursor] = row;
    a[cursor++] = 3.0 + (row % 11) * 0.001;
    if (row + 1 != kN) {
      colidx[cursor] = row + 1;
      a[cursor++] = -0.25;
    }
    rowstr[row + 1] = cursor;
    x[row] = 0.5 + (row % 19) * 0.03125;
  }
  if (cursor != kNnz)
    return 101;

  double rnorm = 0.0;
  for (int iteration = 0; iteration < kWarmups + kSamples; ++iteration) {
    double begin = wall_ms();
    npb_cg_conj_grad_core(kN, kNnz, colidx, rowstr, x, z, a, p, q, r,
                          &rnorm);
    double elapsed = wall_ms() - begin;
    printf("BENCH_SAMPLE phase=%s index=%d wall_ms=%.6f\n",
           iteration < kWarmups ? "warmup" : "sample",
           iteration < kWarmups ? iteration : iteration - kWarmups, elapsed);
  }

  double expected_rnorm = 0.0;
  npb_cg_reference(kN, kNnz, colidx, rowstr, x, expected_z, a, expected_p,
                   expected_q, expected_r, &expected_rnorm);
  double max_abs = fabs(rnorm - expected_rnorm), max_rel = 0.0;
  int ok = isfinite(rnorm);
  for (int i = 0; i < kN; ++i) {
    double z_abs = fabs(z[i] - expected_z[i]);
    double r_abs = fabs(r[i] - expected_r[i]);
    double z_rel = z_abs / fmax(1.0, fabs(expected_z[i]));
    double r_rel = r_abs / fmax(1.0, fabs(expected_r[i]));
    max_abs = fmax(max_abs, fmax(z_abs, r_abs));
    max_rel = fmax(max_rel, fmax(z_rel, r_rel));
    ok = ok && isfinite(z[i]) && isfinite(r[i]);
  }
  ok = ok && (max_abs <= 1.0e-9 || max_rel <= 1.0e-9);
  printf("BENCH_CORRECTNESS max_abs=%.17g max_rel=%.17g atol=1e-9 rtol=1e-9\n",
         max_abs, max_rel);
  printf("BENCH_RESULT workload=npb_cg_conj_grad n=%d nnz=%d status=%s\n",
         kN, kNnz, ok ? "PASS" : "FAIL");

  free(colidx);
  free(rowstr);
  free(a);
  free(x);
  free(z);
  free(p);
  free(q);
  free(r);
  free(expected_z);
  free(expected_p);
  free(expected_q);
  free(expected_r);
  return ok ? 0 : 1;
}
