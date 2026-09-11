// Loop result passed as argument to another function.
extern void sink(double);
void log_sum(int n, double *x) {
  double s = 0;
  for (int i = 0; i < n; i++) s += x[i];
  sink(s);
}
