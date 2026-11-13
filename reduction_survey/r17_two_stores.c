// Reduction stored into two different memrefs.
void duplicate_sum(int n, double *x, double *a, double *b) {
  double s = 0;
  for (int i = 0; i < n; i++) s += x[i];
  *a = s;
  *b = s;
}
