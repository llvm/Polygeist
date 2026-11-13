#include <math.h>

// Source-faithful computational core of NPB3.3-SER-C CG/conj_grad.  The
// original routine obtains these extents from file-static partition globals;
// exposing them as arguments lets the independently linked harness exercise
// the same two CSR loop nests without duplicating those private globals.
// polygeist-arg-extents npb_cg_conj_grad_core: colidx=nnz, rowstr=(n + 1), x=n, z=n, a=nnz, p=n, q=n, r=n, rnorm=1
void npb_cg_conj_grad_core(int n, int nnz, int *colidx, int *rowstr,
                           double *x, double *z, double *a, double *p,
                           double *q, double *r, double *rnorm) {
  int j, k;
  int cgit, cgitmax = 25;
  double d, sum, rho, rho0, alpha, beta;

  rho = 0.0;
  for (j = 0; j < n; j++) {
    q[j] = 0.0;
    z[j] = 0.0;
    r[j] = x[j];
    p[j] = r[j];
  }
  for (j = 0; j < n; j++)
    rho = rho + r[j] * r[j];

  for (cgit = 1; cgit <= cgitmax; cgit++) {
    for (j = 0; j < n; j++) {
      sum = 0.0;
      for (k = rowstr[j]; k < rowstr[j + 1]; k++)
        sum = sum + a[k] * p[colidx[k]];
      q[j] = sum;
    }

    d = 0.0;
    for (j = 0; j < n; j++)
      d = d + p[j] * q[j];
    alpha = rho / d;
    rho0 = rho;
    rho = 0.0;
    for (j = 0; j < n; j++) {
      z[j] = z[j] + alpha * p[j];
      r[j] = r[j] - alpha * q[j];
    }
    for (j = 0; j < n; j++)
      rho = rho + r[j] * r[j];
    beta = rho / rho0;
    for (j = 0; j < n; j++)
      p[j] = r[j] + beta * p[j];
  }

  sum = 0.0;
  for (j = 0; j < n; j++) {
    d = 0.0;
    for (k = rowstr[j]; k < rowstr[j + 1]; k++)
      d = d + a[k] * z[colidx[k]];
    r[j] = d;
  }
  for (j = 0; j < n; j++) {
    d = x[j] - r[j];
    sum = sum + d * d;
  }
  *rnorm = sqrt(sum);
}
