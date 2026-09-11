// Reduction result never used (loop kept for side effects of the body... here, none).
void dead_sum(int n, double *x) {
  double s = 0;
  for (int i = 0; i < n; i++) s += x[i];
}
