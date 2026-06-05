// HPGMG-style mini V-cycle fixture designed to raise cleanly.
//
// This keeps representative stencil, transfer, and BLAS1 loop families while
// avoiding solver structs, MPI state, dynamically dispatched operators, and
// pointer-rich grid metadata.

#define HX 12
#define HY 10
#define HZ 8
#define HV 960

void hpgmg_pipeline_easy(double x[HX + 2][HY + 2][HZ + 2],
                         const double rhs[HX][HY][HZ],
                         const double dinv[HX][HY][HZ],
                         const double fine[2 * HX][2 * HY][2 * HZ],
                         double Ax[HX][HY][HZ], double res[HX][HY][HZ],
                         double next[HX][HY][HZ],
                         double coarse[HX][HY][HZ],
                         double prolong[2 * HX][2 * HY][2 * HZ],
                         double vx[HV], double vr[HV], double vp[HV],
                         const double vAp[HV], double a, double b,
                         double weight, double alpha, double beta) {
  // Apply 7-point operator.
  for (int i = 0; i < HX; i++)
    for (int j = 0; j < HY; j++)
      for (int k = 0; k < HZ; k++) {
        double center = x[i + 1][j + 1][k + 1];
        double lap = 6.0 * center - x[i][j + 1][k + 1] -
                     x[i + 2][j + 1][k + 1] - x[i + 1][j][k + 1] -
                     x[i + 1][j + 2][k + 1] - x[i + 1][j + 1][k] -
                     x[i + 1][j + 1][k + 2];
        Ax[i][j][k] = a * center + b * lap;
      }

  // Residual.
  for (int i = 0; i < HX; i++)
    for (int j = 0; j < HY; j++)
      for (int k = 0; k < HZ; k++) {
        double center = x[i + 1][j + 1][k + 1];
        double lap = 6.0 * center - x[i][j + 1][k + 1] -
                     x[i + 2][j + 1][k + 1] - x[i + 1][j][k + 1] -
                     x[i + 1][j + 2][k + 1] - x[i + 1][j + 1][k] -
                     x[i + 1][j + 1][k + 2];
        res[i][j][k] = rhs[i][j][k] - (a * center + b * lap);
      }

  // Weighted Jacobi smoother.
  for (int i = 0; i < HX; i++)
    for (int j = 0; j < HY; j++)
      for (int k = 0; k < HZ; k++) {
        double center = x[i + 1][j + 1][k + 1];
        double lap = 6.0 * center - x[i][j + 1][k + 1] -
                     x[i + 2][j + 1][k + 1] - x[i + 1][j][k + 1] -
                     x[i + 1][j + 2][k + 1] - x[i + 1][j + 1][k] -
                     x[i + 1][j + 1][k + 2];
        double ax = a * center + b * lap;
        next[i][j][k] = center + weight * dinv[i][j][k] * (rhs[i][j][k] - ax);
      }

  // Cell-centered restriction.
  for (int i = 0; i < HX; i++)
    for (int j = 0; j < HY; j++)
      for (int k = 0; k < HZ; k++) {
        int ii = 2 * i, jj = 2 * j, kk = 2 * k;
        coarse[i][j][k] =
            (fine[ii][jj][kk] + fine[ii + 1][jj][kk] +
             fine[ii][jj + 1][kk] + fine[ii + 1][jj + 1][kk] +
             fine[ii][jj][kk + 1] + fine[ii + 1][jj][kk + 1] +
             fine[ii][jj + 1][kk + 1] + fine[ii + 1][jj + 1][kk + 1]) *
            0.125;
      }

  // Simple injection/prolongation.
  for (int i = 0; i < HX; i++)
    for (int j = 0; j < HY; j++)
      for (int k = 0; k < HZ; k++)
        prolong[2 * i][2 * j][2 * k] = coarse[i][j][k];

  // BLAS1-style CG update.
  for (int i = 0; i < HV; i++) {
    vx[i] += alpha * vp[i];
    vr[i] -= alpha * vAp[i];
    vp[i] = vr[i] + beta * vp[i];
  }
}
