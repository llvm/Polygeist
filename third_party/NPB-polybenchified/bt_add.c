// PolyBench-style extraction of NPB BT's `add` kernel.
// Original (NPB3.0-omp-C/BT/bt.c lines 181-199): u[i][j][k][m] += rhs[i][j][k][m]
// over the interior of the 4D field.
//
// In NPB, `u` and `rhs` are file-local static 4D arrays, and `grid_points` is
// a 3-element static int array set at runtime. Here we pass them as parameters
// with class-S sizes (problem_size = 12 ⇒ IMAX = JMAX = KMAX = 12 + 1).

#define IMAX 13
#define JMAX 13
#define KMAX 13

// Bounds passed as scalar ints (not loaded from an array) so the raise pass
// can recognise the loops as affine.
void bt_add(double u[IMAX][JMAX][KMAX][5],
            double rhs[IMAX][JMAX][KMAX][5],
            int gpx, int gpy, int gpz) {
  int i, j, k, m;

  for (i = 1; i < gpx - 1; i++) {
    for (j = 1; j < gpy - 1; j++) {
      for (k = 1; k < gpz - 1; k++) {
        for (m = 0; m < 5; m++) {
          u[i][j][k][m] = u[i][j][k][m] + rhs[i][j][k][m];
        }
      }
    }
  }
}
