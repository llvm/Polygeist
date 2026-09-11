// Reduction result used as an upper-bound expression for a later loop.
extern void use_int(int);
void hist(int n, double *x) {
  int c = 0;
  for (int i = 0; i < n; i++) if (x[i] > 0) c++;
  for (int j = 0; j < c; j++) use_int(j);
}
