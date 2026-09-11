// PolyBench-style extraction of NPB LU's `l2norm` kernel.
// Original (NPB3.0-omp-C/LU/lu.c lines 1981-2030).
// Computes the 5-component L2 norm of a 4D field v over the interior.
//
// NPB pads dims 2 and 3 by 1 ("ISIZ2/2*2+1") — we keep that exactly so the
// access pattern matches.

#define ISIZ1 12
#define ISIZ2 12
#define ISIZ3 12
#define D2 (ISIZ2/2*2 + 1)
#define D3 (ISIZ3/2*2 + 1)

void lu_l2norm(int nx0, int ny0, int nz0,
               int ist, int iend,
               int jst, int jend,
               double v[ISIZ1][D2][D3][5],
               double sum[5]) {
  int i, j, k, m;

  for (m = 0; m < 5; m++) sum[m] = 0.0;

  for (i = ist; i <= iend; i++) {
    for (j = jst; j <= jend; j++) {
      for (k = 1; k <= nz0 - 2; k++) {
        sum[0] = sum[0] + v[i][j][k][0] * v[i][j][k][0];
        sum[1] = sum[1] + v[i][j][k][1] * v[i][j][k][1];
        sum[2] = sum[2] + v[i][j][k][2] * v[i][j][k][2];
        sum[3] = sum[3] + v[i][j][k][3] * v[i][j][k][3];
        sum[4] = sum[4] + v[i][j][k][4] * v[i][j][k][4];
      }
    }
  }
}
