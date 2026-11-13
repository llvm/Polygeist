// PolyBench-style extraction of NPB FT's `evolve` kernel.
// Original (NPB3.0-omp-C/FT/ft.c lines 225-245): u1 = u0 * ex[t*indexmap].
//
// The original uses a `dcomplex` struct {double real; double imag;}; we
// flatten that to a trailing dimension of size 2 so the IR sees a plain
// rank-4 double array — exactly how cgeist would lower the struct anyway.

#define NX 64
#define NY 64
#define NZ 64
#define EXP_MAX (200 * (NX*NX/4 + NY*NY/4 + NZ*NZ/4))

// d-dimensions passed as scalar ints so the loops are recognised as affine.
void ft_evolve(double u0[NZ][NY][NX][2],
               double u1[NZ][NY][NX][2],
               int t,
               int indexmap[NZ][NY][NX],
               int d0, int d1, int d2,
               double ex[EXP_MAX]) {
  int i, j, k;
  for (k = 0; k < d2; k++) {
    for (j = 0; j < d1; j++) {
      for (i = 0; i < d0; i++) {
        double scale = ex[t * indexmap[k][j][i]];
        u1[k][j][i][0] = u0[k][j][i][0] * scale;
        u1[k][j][i][1] = u0[k][j][i][1] * scale;
      }
    }
  }
}
