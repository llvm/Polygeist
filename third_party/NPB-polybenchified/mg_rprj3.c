// PolyBench-style extraction of NPB MG's `rprj3` kernel (restriction operator).
// Original (NPB3.0-omp-C/MG/mg.c lines 557-636): projects a fine-grid array r
// onto a coarse-grid s via trilinear FE projection (s = P r). Loops over the
// coarse grid; reads at i = 2*j - d (downsampling).
//
// The `d1/d2/d3` step factors depend on whether the coarse grid dim equals 3
// (boundary case). We pass them as scalars.

// Fine-grid size N1f x N2f x N3f, coarse-grid size N1c x N2c x N3c.
#define N1F 34
#define N2F 34
#define N3F 34
#define N1C 18
#define N2C 18
#define N3C 18
#define M   35

void mg_rprj3(double r[N3F][N2F][N1F], int m1k, int m2k, int m3k,
              double s[N3C][N2C][N1C], int m1j, int m2j, int m3j,
              int d1, int d2, int d3) {
  int j3, j2, j1, i3, i2, i1;
  double x1[M], y1[M], x2, y2;

  for (j3 = 1; j3 < m3j - 1; j3++) {
    i3 = 2 * j3 - d3;
    for (j2 = 1; j2 < m2j - 1; j2++) {
      i2 = 2 * j2 - d2;

      for (j1 = 1; j1 < m1j; j1++) {
        i1 = 2 * j1 - d1;
        x1[i1] = r[i3+1][i2][i1] + r[i3+1][i2+2][i1]
               + r[i3][i2+1][i1] + r[i3+2][i2+1][i1];
        y1[i1] = r[i3][i2][i1] + r[i3+2][i2][i1]
               + r[i3][i2+2][i1] + r[i3+2][i2+2][i1];
      }

      for (j1 = 1; j1 < m1j - 1; j1++) {
        i1 = 2 * j1 - d1;
        y2 = r[i3][i2][i1+1] + r[i3+2][i2][i1+1]
           + r[i3][i2+2][i1+1] + r[i3+2][i2+2][i1+1];
        x2 = r[i3+1][i2][i1+1] + r[i3+1][i2+2][i1+1]
           + r[i3][i2+1][i1+1] + r[i3+2][i2+1][i1+1];
        s[j3][j2][j1] =
            0.5    * r[i3+1][i2+1][i1+1]
          + 0.25   * ( r[i3+1][i2+1][i1] + r[i3+1][i2+1][i1+2] + x2)
          + 0.125  * ( x1[i1] + x1[i1+2] + y2)
          + 0.0625 * ( y1[i1] + y1[i1+2] );
      }
    }
  }
}
