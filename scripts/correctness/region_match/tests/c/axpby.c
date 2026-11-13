/* Positive (composition): reference BLAS has no AXPBY; y = alpha*x + beta*y
 * is DSCAL(beta, y) followed by DAXPY(alpha, x, y). */
void axpby(int n, double alpha, double beta, double x[n], double y[n]) {
  for (int i = 0; i < n; i++)
    y[i] = alpha * x[i] + beta * y[i];
}
