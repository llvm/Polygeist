/* Negative (composition): the scaled term reads the neighbour y[i+1]. */
void axpby_shift(int n, double alpha, double beta, double x[n], double y[n]) {
  for (int i = 0; i < n - 1; i++)
    y[i] = alpha * x[i] + beta * y[i + 1];
}
