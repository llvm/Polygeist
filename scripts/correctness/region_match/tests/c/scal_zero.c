/* Negative (literal policy, both matchers): x = 0.0 * x must not become DSCAL
 * with da = 0.0. */
void scal_zero(int n, double x[n]) {
  for (int i = 0; i < n; i++)
    x[i] = 0.0 * x[i];
}
