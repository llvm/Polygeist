/* Concrete C extraction of MFEM's symmetric PA diffusion apply kernels. */
enum { D1D = 4, Q1D = 5, NE = 2 };

#define X2(dx, dy, e) ((dx) + D1D * ((dy) + D1D * (e)))
#define X3(dx, dy, dz, e) \
  ((dx) + D1D * ((dy) + D1D * ((dz) + D1D * (e))))
#define OP2(q, c, e) ((q) + Q1D * Q1D * ((c) + 3 * (e)))
#define OP3(q, c, e) ((q) + Q1D * Q1D * Q1D * ((c) + 6 * (e)))

void mfem_pa_diffusion_apply_2d(const double *B, const double *G,
                                const double *Bt, const double *Gt,
                                const double *D, const double *X, double *Y) {
  for (int e = 0; e < NE; ++e) {
    double grad[Q1D][Q1D][2];
    for (int qy = 0; qy < Q1D; ++qy)
      for (int qx = 0; qx < Q1D; ++qx)
        grad[qy][qx][0] = grad[qy][qx][1] = 0.0;
    for (int dy = 0; dy < D1D; ++dy) {
      double grad_x[Q1D][2];
      for (int qx = 0; qx < Q1D; ++qx)
        grad_x[qx][0] = grad_x[qx][1] = 0.0;
      for (int dx = 0; dx < D1D; ++dx) {
        double s = X[X2(dx, dy, e)];
        for (int qx = 0; qx < Q1D; ++qx) {
          grad_x[qx][0] += s * B[qx * D1D + dx];
          grad_x[qx][1] += s * G[qx * D1D + dx];
        }
      }
      for (int qy = 0; qy < Q1D; ++qy)
        for (int qx = 0; qx < Q1D; ++qx) {
          grad[qy][qx][0] += grad_x[qx][1] * B[qy * D1D + dy];
          grad[qy][qx][1] += grad_x[qx][0] * G[qy * D1D + dy];
        }
    }
    for (int qy = 0; qy < Q1D; ++qy)
      for (int qx = 0; qx < Q1D; ++qx) {
        int q = qx + Q1D * qy;
        double gx = grad[qy][qx][0], gy = grad[qy][qx][1];
        double o11 = D[OP2(q, 0, e)], o12 = D[OP2(q, 1, e)];
        double o22 = D[OP2(q, 2, e)];
        grad[qy][qx][0] = o11 * gx + o12 * gy;
        grad[qy][qx][1] = o12 * gx + o22 * gy;
      }
    for (int qy = 0; qy < Q1D; ++qy) {
      double grad_x[D1D][2];
      for (int dx = 0; dx < D1D; ++dx)
        grad_x[dx][0] = grad_x[dx][1] = 0.0;
      for (int qx = 0; qx < Q1D; ++qx)
        for (int dx = 0; dx < D1D; ++dx) {
          grad_x[dx][0] += grad[qy][qx][0] * Gt[dx * Q1D + qx];
          grad_x[dx][1] += grad[qy][qx][1] * Bt[dx * Q1D + qx];
        }
      for (int dy = 0; dy < D1D; ++dy)
        for (int dx = 0; dx < D1D; ++dx)
          Y[X2(dx, dy, e)] += grad_x[dx][0] * Bt[dy * Q1D + qy]
                            + grad_x[dx][1] * Gt[dy * Q1D + qy];
    }
  }
}

void mfem_pa_diffusion_apply_3d(const double *B, const double *G,
                                const double *Bt, const double *Gt,
                                const double *D, const double *X, double *Y) {
  for (int e = 0; e < NE; ++e) {
    double grad[Q1D][Q1D][Q1D][3];
    for (int qz = 0; qz < Q1D; ++qz)
      for (int qy = 0; qy < Q1D; ++qy)
        for (int qx = 0; qx < Q1D; ++qx)
          for (int c = 0; c < 3; ++c) grad[qz][qy][qx][c] = 0.0;
    for (int dz = 0; dz < D1D; ++dz) {
      double grad_xy[Q1D][Q1D][3];
      for (int qy = 0; qy < Q1D; ++qy)
        for (int qx = 0; qx < Q1D; ++qx)
          for (int c = 0; c < 3; ++c) grad_xy[qy][qx][c] = 0.0;
      for (int dy = 0; dy < D1D; ++dy) {
        double grad_x[Q1D][2];
        for (int qx = 0; qx < Q1D; ++qx)
          grad_x[qx][0] = grad_x[qx][1] = 0.0;
        for (int dx = 0; dx < D1D; ++dx) {
          double s = X[X3(dx, dy, dz, e)];
          for (int qx = 0; qx < Q1D; ++qx) {
            grad_x[qx][0] += s * B[qx * D1D + dx];
            grad_x[qx][1] += s * G[qx * D1D + dx];
          }
        }
        for (int qy = 0; qy < Q1D; ++qy)
          for (int qx = 0; qx < Q1D; ++qx) {
            grad_xy[qy][qx][0] += grad_x[qx][1] * B[qy * D1D + dy];
            grad_xy[qy][qx][1] += grad_x[qx][0] * G[qy * D1D + dy];
            grad_xy[qy][qx][2] += grad_x[qx][0] * B[qy * D1D + dy];
          }
      }
      for (int qz = 0; qz < Q1D; ++qz)
        for (int qy = 0; qy < Q1D; ++qy)
          for (int qx = 0; qx < Q1D; ++qx) {
            grad[qz][qy][qx][0] += grad_xy[qy][qx][0] * B[qz * D1D + dz];
            grad[qz][qy][qx][1] += grad_xy[qy][qx][1] * B[qz * D1D + dz];
            grad[qz][qy][qx][2] += grad_xy[qy][qx][2] * G[qz * D1D + dz];
          }
    }
    for (int qz = 0; qz < Q1D; ++qz)
      for (int qy = 0; qy < Q1D; ++qy)
        for (int qx = 0; qx < Q1D; ++qx) {
          int q = qx + Q1D * (qy + Q1D * qz);
          double x = grad[qz][qy][qx][0], y = grad[qz][qy][qx][1];
          double z = grad[qz][qy][qx][2];
          double o11 = D[OP3(q, 0, e)], o12 = D[OP3(q, 1, e)];
          double o13 = D[OP3(q, 2, e)], o22 = D[OP3(q, 3, e)];
          double o23 = D[OP3(q, 4, e)], o33 = D[OP3(q, 5, e)];
          grad[qz][qy][qx][0] = o11*x + o12*y + o13*z;
          grad[qz][qy][qx][1] = o12*x + o22*y + o23*z;
          grad[qz][qy][qx][2] = o13*x + o23*y + o33*z;
        }
    for (int qz = 0; qz < Q1D; ++qz) {
      double grad_xy[D1D][D1D][3];
      for (int dy = 0; dy < D1D; ++dy)
        for (int dx = 0; dx < D1D; ++dx)
          for (int c = 0; c < 3; ++c) grad_xy[dy][dx][c] = 0.0;
      for (int qy = 0; qy < Q1D; ++qy) {
        double grad_x[D1D][3];
        for (int dx = 0; dx < D1D; ++dx)
          for (int c = 0; c < 3; ++c) grad_x[dx][c] = 0.0;
        for (int qx = 0; qx < Q1D; ++qx)
          for (int dx = 0; dx < D1D; ++dx) {
            grad_x[dx][0] += grad[qz][qy][qx][0] * Gt[dx * Q1D + qx];
            grad_x[dx][1] += grad[qz][qy][qx][1] * Bt[dx * Q1D + qx];
            grad_x[dx][2] += grad[qz][qy][qx][2] * Bt[dx * Q1D + qx];
          }
        for (int dy = 0; dy < D1D; ++dy)
          for (int dx = 0; dx < D1D; ++dx) {
            grad_xy[dy][dx][0] += grad_x[dx][0] * Bt[dy * Q1D + qy];
            grad_xy[dy][dx][1] += grad_x[dx][1] * Gt[dy * Q1D + qy];
            grad_xy[dy][dx][2] += grad_x[dx][2] * Bt[dy * Q1D + qy];
          }
      }
      for (int dz = 0; dz < D1D; ++dz)
        for (int dy = 0; dy < D1D; ++dy)
          for (int dx = 0; dx < D1D; ++dx)
            Y[X3(dx, dy, dz, e)] +=
                (grad_xy[dy][dx][0] + grad_xy[dy][dx][1]) * Bt[dz * Q1D + qz]
              + grad_xy[dy][dx][2] * Gt[dz * Q1D + qz];
    }
  }
}
