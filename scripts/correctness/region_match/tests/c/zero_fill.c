/* Negative (literal policy): y = 0 must not become DSCAL with da = 0.0
 * (0 * NaN = NaN, the store of 0 is not). */
void zero_fill(int n, double y[n]) {
  for (int i = 0; i < n; i++)
    y[i] = 0.0;
}
