/* Concrete C extraction of MFEM partial-assembly mass apply kernels. */
enum { D1D = 4, Q1D = 5, NE = 2 };

#define X2(dx, dy, e) ((dx) + D1D * ((dy) + D1D * (e)))
#define D2(qx, qy, e) ((qx) + Q1D * ((qy) + Q1D * (e)))
#define X3(dx, dy, dz, e) \
  ((dx) + D1D * ((dy) + D1D * ((dz) + D1D * (e))))
#define D3(qx, qy, qz, e) \
  ((qx) + Q1D * ((qy) + Q1D * ((qz) + Q1D * (e))))

void mfem_pa_mass_apply_2d(const double *B, const double *Bt,
                           const double *D, const double *X, double *Y) {
  for (int e = 0; e < NE; ++e) {
    double sol_xy[Q1D][Q1D];
    for (int qy = 0; qy < Q1D; ++qy)
      for (int qx = 0; qx < Q1D; ++qx)
        sol_xy[qy][qx] = 0.0;
    for (int dy = 0; dy < D1D; ++dy) {
      double sol_x[Q1D];
      for (int qx = 0; qx < Q1D; ++qx)
        sol_x[qx] = 0.0;
      for (int dx = 0; dx < D1D; ++dx) {
        double s = X[X2(dx, dy, e)];
        for (int qx = 0; qx < Q1D; ++qx)
          sol_x[qx] += B[qx * D1D + dx] * s;
      }
      for (int qy = 0; qy < Q1D; ++qy)
        for (int qx = 0; qx < Q1D; ++qx)
          sol_xy[qy][qx] += B[qy * D1D + dy] * sol_x[qx];
    }
    for (int qy = 0; qy < Q1D; ++qy)
      for (int qx = 0; qx < Q1D; ++qx)
        sol_xy[qy][qx] *= D[D2(qx, qy, e)];
    for (int qy = 0; qy < Q1D; ++qy) {
      double sol_x[D1D];
      for (int dx = 0; dx < D1D; ++dx)
        sol_x[dx] = 0.0;
      for (int qx = 0; qx < Q1D; ++qx)
        for (int dx = 0; dx < D1D; ++dx)
          sol_x[dx] += Bt[dx * Q1D + qx] * sol_xy[qy][qx];
      for (int dy = 0; dy < D1D; ++dy)
        for (int dx = 0; dx < D1D; ++dx)
          Y[X2(dx, dy, e)] += Bt[dy * Q1D + qy] * sol_x[dx];
    }
  }
}

void mfem_pa_mass_apply_3d(const double *B, const double *Bt,
                           const double *D, const double *X, double *Y) {
  for (int e = 0; e < NE; ++e) {
    double sol_xyz[Q1D][Q1D][Q1D];
    for (int qz = 0; qz < Q1D; ++qz)
      for (int qy = 0; qy < Q1D; ++qy)
        for (int qx = 0; qx < Q1D; ++qx)
          sol_xyz[qz][qy][qx] = 0.0;
    for (int dz = 0; dz < D1D; ++dz) {
      double sol_xy[Q1D][Q1D];
      for (int qy = 0; qy < Q1D; ++qy)
        for (int qx = 0; qx < Q1D; ++qx)
          sol_xy[qy][qx] = 0.0;
      for (int dy = 0; dy < D1D; ++dy) {
        double sol_x[Q1D];
        for (int qx = 0; qx < Q1D; ++qx)
          sol_x[qx] = 0.0;
        for (int dx = 0; dx < D1D; ++dx)
          for (int qx = 0; qx < Q1D; ++qx)
            sol_x[qx] += B[qx * D1D + dx] * X[X3(dx, dy, dz, e)];
        for (int qy = 0; qy < Q1D; ++qy)
          for (int qx = 0; qx < Q1D; ++qx)
            sol_xy[qy][qx] += B[qy * D1D + dy] * sol_x[qx];
      }
      for (int qz = 0; qz < Q1D; ++qz)
        for (int qy = 0; qy < Q1D; ++qy)
          for (int qx = 0; qx < Q1D; ++qx)
            sol_xyz[qz][qy][qx] += B[qz * D1D + dz] * sol_xy[qy][qx];
    }
    for (int qz = 0; qz < Q1D; ++qz)
      for (int qy = 0; qy < Q1D; ++qy)
        for (int qx = 0; qx < Q1D; ++qx)
          sol_xyz[qz][qy][qx] *= D[D3(qx, qy, qz, e)];
    for (int qz = 0; qz < Q1D; ++qz) {
      double sol_xy[D1D][D1D];
      for (int dy = 0; dy < D1D; ++dy)
        for (int dx = 0; dx < D1D; ++dx)
          sol_xy[dy][dx] = 0.0;
      for (int qy = 0; qy < Q1D; ++qy) {
        double sol_x[D1D];
        for (int dx = 0; dx < D1D; ++dx)
          sol_x[dx] = 0.0;
        for (int qx = 0; qx < Q1D; ++qx)
          for (int dx = 0; dx < D1D; ++dx)
            sol_x[dx] += Bt[dx * Q1D + qx] * sol_xyz[qz][qy][qx];
        for (int dy = 0; dy < D1D; ++dy)
          for (int dx = 0; dx < D1D; ++dx)
            sol_xy[dy][dx] += Bt[dy * Q1D + qy] * sol_x[dx];
      }
      for (int dz = 0; dz < D1D; ++dz)
        for (int dy = 0; dy < D1D; ++dy)
          for (int dx = 0; dx < D1D; ++dx)
            Y[X3(dx, dy, dz, e)] += Bt[dz * Q1D + qz] * sol_xy[dy][dx];
    }
  }
}
