// Source-faithful computational cores extracted from NPB3.3-SER-C/MG/mg.c.
// Boundary exchange, timers, and debug printing are deliberately outside the
// idiom under test.  Class-S uses M=19; test dimensions must not exceed it.
#define M 19

void resid(signed char ou[1], signed char ov[1], signed char orr[1],
           int n1, int n2, int n3, double a[4], int k) {
  double (*u)[n2][n1] = (double (*)[n2][n1])ou;
  double (*v)[n2][n1] = (double (*)[n2][n1])ov;
  double (*r)[n2][n1] = (double (*)[n2][n1])orr;
  double u1[M], u2[M];
  (void)k;
  for (int i3 = 1; i3 < n3 - 1; i3++) {
    for (int i2 = 1; i2 < n2 - 1; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        u1[i1] = u[i3][i2-1][i1] + u[i3][i2+1][i1]
               + u[i3-1][i2][i1] + u[i3+1][i2][i1];
        u2[i1] = u[i3-1][i2-1][i1] + u[i3-1][i2+1][i1]
               + u[i3+1][i2-1][i1] + u[i3+1][i2+1][i1];
      }
      for (int i1 = 1; i1 < n1 - 1; i1++) {
        r[i3][i2][i1] = v[i3][i2][i1]
                      - a[0] * u[i3][i2][i1]
                      - a[2] * (u2[i1] + u1[i1-1] + u1[i1+1])
                      - a[3] * (u2[i1-1] + u2[i1+1]);
      }
    }
  }
}

void psinv(signed char orr[1], signed char ou[1],
           int n1, int n2, int n3, double c[4], int k) {
  double (*r)[n2][n1] = (double (*)[n2][n1])orr;
  double (*u)[n2][n1] = (double (*)[n2][n1])ou;
  double r1[M], r2[M];
  (void)k;
  for (int i3 = 1; i3 < n3 - 1; i3++) {
    for (int i2 = 1; i2 < n2 - 1; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        r1[i1] = r[i3][i2-1][i1] + r[i3][i2+1][i1]
               + r[i3-1][i2][i1] + r[i3+1][i2][i1];
        r2[i1] = r[i3-1][i2-1][i1] + r[i3-1][i2+1][i1]
               + r[i3+1][i2-1][i1] + r[i3+1][i2+1][i1];
      }
      for (int i1 = 1; i1 < n1 - 1; i1++) {
        u[i3][i2][i1] = u[i3][i2][i1]
                      + c[0] * r[i3][i2][i1]
                      + c[1] * (r[i3][i2][i1-1] + r[i3][i2][i1+1]
                              + r1[i1])
                      + c[2] * (r2[i1] + r1[i1-1] + r1[i1+1]);
      }
    }
  }
}
