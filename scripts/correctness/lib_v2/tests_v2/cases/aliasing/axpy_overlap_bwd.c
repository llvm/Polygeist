//@ kind: explicit
//@ expect: reject_or_correct
// daxpy(n-1, alpha, x+1, x): every x[i+1] is read before it is written, so
// this IS daxpy on the original values (a match is legal if it copies x+1).
void axpy_overlap_bwd(int n, double alpha, double x[n]) {
  for (int i = 0; i < n - 1; i++)
    x[i] += alpha * x[i + 1];
}
