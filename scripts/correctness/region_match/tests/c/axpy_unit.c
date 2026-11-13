/* Positive (literal binding): y += x has no alpha; DAXPY's da binds to the
 * literal 1.0 (e-graph union with Term.Lit). */
void axpy_unit(int n, double x[n], double y[n]) {
  for (int i = 0; i < n; i++)
    y[i] = y[i] + x[i];
}
