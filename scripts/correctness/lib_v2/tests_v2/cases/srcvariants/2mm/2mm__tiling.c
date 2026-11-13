//@ transform: tiling
//@ function: mm2__tiling
// i strip-mined by 8 in both nests.
void mm2__tiling(int ni, int nj, int nk, int nl, double alpha, double beta, double tmp[ni][nj], double A[ni][nk], double B[nk][nj], double C[nj][nl], double D[ni][nl]) {
  for (int ii = 0; ii < ni; ii += 8) {
    int ie = ii + 8 < ni ? ii + 8 : ni;
    for (int i = ii; i < ie; i++)
      for (int j = 0; j < nj; j++) {
        tmp[i][j] = 0.0;
        for (int k = 0; k < nk; ++k)
          tmp[i][j] += alpha * A[i][k] * B[k][j];
      }
  }
  for (int ii = 0; ii < ni; ii += 8) {
    int ie = ii + 8 < ni ? ii + 8 : ni;
    for (int i = ii; i < ie; i++)
      for (int j = 0; j < nl; j++) {
        D[i][j] *= beta;
        for (int k = 0; k < nj; ++k)
          D[i][j] += tmp[i][k] * C[k][j];
      }
  }
}
