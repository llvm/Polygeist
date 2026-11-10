/* Negative (composition): a product of the two scaled terms is no
 * DSCAL/DAXPY sequence. */
void axpby_prod(int n, double alpha, double beta, double x[n], double y[n]) {
  for (int i = 0; i < n; i++)
    y[i] = (alpha * x[i]) * (beta * y[i]);
}
