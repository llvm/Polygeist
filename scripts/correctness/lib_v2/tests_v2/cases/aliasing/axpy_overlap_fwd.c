//@ kind: explicit
//@ expect: reject_or_correct
// daxpy(n-1, alpha, x, x+1): y overlaps x shifted by +1 -> a recurrence.
void axpy_overlap_fwd(int n, double alpha, double x[n]) {
  for (int i = 0; i < n - 1; i++)
    x[i + 1] += alpha * x[i];
}
