// PolyBench-style extraction of NPB MG's `resid` kernel (residual r = v - Au).
// Original (NPB3.0-omp-C/MG/mg.c lines 495-552).
//
// Same shape as psinv (27-point stencil via two scratch rows) but writes r
// instead of u and uses coefficients a[0]..a[3] (with a[1]=0 elided).

#define N1 34
#define N2 34
#define N3 34
#define M  35

void mg_resid(double u[N3][N2][N1],
              double v[N3][N2][N1],
              double r[N3][N2][N1],
              int n1, int n2, int n3,
              double a[4]) {
  int i3, i2, i1;
  double u1[M], u2[M];

  for (i3 = 1; i3 < n3 - 1; i3++) {
    for (i2 = 1; i2 < n2 - 1; i2++) {
      for (i1 = 0; i1 < n1; i1++) {
        u1[i1] = u[i3][i2-1][i1] + u[i3][i2+1][i1]
               + u[i3-1][i2][i1] + u[i3+1][i2][i1];
        u2[i1] = u[i3-1][i2-1][i1] + u[i3-1][i2+1][i1]
               + u[i3+1][i2-1][i1] + u[i3+1][i2+1][i1];
      }
      for (i1 = 1; i1 < n1 - 1; i1++) {
        r[i3][i2][i1] = v[i3][i2][i1]
            - a[0] * u[i3][i2][i1]
            - a[2] * ( u2[i1] + u1[i1-1] + u1[i1+1] )
            - a[3] * ( u2[i1-1] + u2[i1+1] );
      }
    }
  }
}
