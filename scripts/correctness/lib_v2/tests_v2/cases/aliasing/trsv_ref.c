//@ kind: runtime_base
//@ init: spd
// PolyBench trisolv in VLA form; used for the runtime b == x aliasing case.
void trsv_ref(int n, double L[n][n], double x[n], double b[n]) {
  for (int i = 0; i < n; i++) {
    x[i] = b[i];
    for (int j = 0; j < i; j++)
      x[i] -= L[i][j] * x[j];
    x[i] = x[i] / L[i][i];
  }
}
