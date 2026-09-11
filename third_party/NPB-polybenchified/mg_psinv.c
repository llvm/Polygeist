// PolyBench-style extraction of NPB MG's `psinv` kernel (smoother).
// Original (NPB3.0-omp-C/MG/mg.c lines 434-490): u = u + Cr, with 27-stencil
// applied via two scratch rows r1[], r2[].
//
// NPB MG uses `double ***` triple-pointer arrays. We rewrite as fixed-size
// 3D `double [N3][N2][N1]` (the polybench convention). N1=N2=N3=34 picks
// class-S MG: lt=8, nx=ny=nz=32, +2 ghost = 34. The kernel itself doesn't
// depend on the exact size; we pass n1/n2/n3 as parameters for the bounds.

#define N1 34
#define N2 34
#define N3 34
#define M  35

void mg_psinv(double r[N3][N2][N1],
              double u[N3][N2][N1],
              int n1, int n2, int n3,
              double c[4]) {
  int i3, i2, i1;
  double r1[M], r2[M];

  for (i3 = 1; i3 < n3 - 1; i3++) {
    for (i2 = 1; i2 < n2 - 1; i2++) {
      for (i1 = 0; i1 < n1; i1++) {
        r1[i1] = r[i3][i2-1][i1] + r[i3][i2+1][i1]
               + r[i3-1][i2][i1] + r[i3+1][i2][i1];
        r2[i1] = r[i3-1][i2-1][i1] + r[i3-1][i2+1][i1]
               + r[i3+1][i2-1][i1] + r[i3+1][i2+1][i1];
      }
      for (i1 = 1; i1 < n1 - 1; i1++) {
        u[i3][i2][i1] = u[i3][i2][i1]
            + c[0] * r[i3][i2][i1]
            + c[1] * ( r[i3][i2][i1-1] + r[i3][i2][i1+1] + r1[i1] )
            + c[2] * ( r2[i1] + r1[i1-1] + r1[i1+1] );
      }
    }
  }
}
