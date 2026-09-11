// Outer iter_arg accumulates inner-loop reduction (nested reductions).
double dist(int m, int n, double *A) {
  double total = 0;
  for (int i = 0; i < m; i++) {
    double row = 0;
    for (int j = 0; j < n; j++) row += A[i*n + j];
    total += row * row;
  }
  return total;
}
