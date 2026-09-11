#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void npb_cg_conj_grad_core(int n, int nnz, int *colidx, int *rowstr,
                           double *x, double *z, double *a, double *p,
                           double *q, double *r, double *rnorm);

static void reference(int n, const int *colidx, const int *rowstr,
                      const double *x, double *z, const double *a, double *p,
                      double *q, double *r, double *rnorm) {
  double rho = 0.0;
  for (int j = 0; j < n; ++j) {
    q[j] = z[j] = 0.0;
    r[j] = p[j] = x[j];
    rho += r[j] * r[j];
  }
  for (int iteration = 0; iteration < 25; ++iteration) {
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int k = rowstr[j]; k < rowstr[j + 1]; ++k)
        sum += a[k] * p[colidx[k]];
      q[j] = sum;
    }
    double dot = 0.0;
    for (int j = 0; j < n; ++j)
      dot += p[j] * q[j];
    double alpha = rho / dot;
    double old_rho = rho;
    rho = 0.0;
    for (int j = 0; j < n; ++j) {
      z[j] += alpha * p[j];
      r[j] -= alpha * q[j];
    }
    for (int j = 0; j < n; ++j)
      rho += r[j] * r[j];
    double beta = rho / old_rho;
    for (int j = 0; j < n; ++j)
      p[j] = r[j] + beta * p[j];
  }
  double sum = 0.0;
  for (int j = 0; j < n; ++j) {
    double value = 0.0;
    for (int k = rowstr[j]; k < rowstr[j + 1]; ++k)
      value += a[k] * z[colidx[k]];
    r[j] = value;
    double residual = x[j] - value;
    sum += residual * residual;
  }
  *rnorm = sqrt(sum);
}

static int close_enough(double got, double expected) {
  double scale = fmax(1.0, fabs(expected));
  return isfinite(got) && fabs(got - expected) <= 1.0e-9 * scale;
}

int main(void) {
  enum { n = 64, nnz = 190 };
  int colidx[nnz], rowstr[n + 1];
  double a[nnz], x[n];
  int cursor = 0;
  rowstr[0] = 0;
  for (int row = 0; row < n; ++row) {
    if (row != 0) {
      colidx[cursor] = row - 1;
      a[cursor++] = -0.37;
    }
    colidx[cursor] = row;
    a[cursor++] = 2.5 + 0.01 * row;
    if (row + 1 != n) {
      colidx[cursor] = row + 1;
      a[cursor++] = -0.37;
    }
    rowstr[row + 1] = cursor;
    x[row] = 0.25 + (double)((row * 17) % 23) / 19.0;
  }
  if (cursor != nnz)
    return 2;

  double z[n], p[n], q[n], r[n], rnorm = 0.0;
  double ez[n], ep[n], eq[n], er[n], expected_rnorm = 0.0;
  reference(n, colidx, rowstr, x, ez, a, ep, eq, er, &expected_rnorm);
  npb_cg_conj_grad_core(n, nnz, colidx, rowstr, x, z, a, p, q, r,
                        &rnorm);

  for (int i = 0; i < n; ++i) {
    if (!close_enough(z[i], ez[i]) || !close_enough(r[i], er[i])) {
      fprintf(stderr,
              "npb-cg-conj-grad-core: FAIL index=%d z=%0.17g expected=%0.17g "
              "r=%0.17g expected-r=%0.17g\n",
              i, z[i], ez[i], r[i], er[i]);
      return 1;
    }
  }
  if (!close_enough(rnorm, expected_rnorm)) {
    fprintf(stderr, "npb-cg-conj-grad-core: FAIL rnorm=%0.17g expected=%0.17g\n",
            rnorm, expected_rnorm);
    return 1;
  }
  puts("npb-cg-conj-grad-core: PASS");
  return 0;
}
