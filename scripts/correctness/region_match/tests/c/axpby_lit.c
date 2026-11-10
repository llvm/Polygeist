/* Positive (composition + literals): y = 2*y + x -> DSCAL(2.0) ; DAXPY(1.0). */
void axpby_lit(int n, double x[n], double y[n]) {
  for (int i = 0; i < n; i++)
    y[i] = 2.0 * y[i] + x[i];
}
