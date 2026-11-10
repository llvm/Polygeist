/* Positive (literal binding): an application constant becomes the scalar. */
void axpy_const(int n, double x[n], double y[n]) {
  for (int i = 0; i < n; i++)
    y[i] = y[i] + 2.5 * x[i];
}
