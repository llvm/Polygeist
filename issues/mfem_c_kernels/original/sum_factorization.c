/*
 * Concrete C extraction of MFEM's tensor-product interpolation/integration.
 * Upstream: fem/dfem/{interpolate,integrate}.hpp at MFEM 951cf8886b9c0c33.
 * The local scratch arrays are intentional: they preserve the staged
 * sum-factorized algorithm and exercise alloca/load/store raising.
 */
enum { D1D = 4, Q1D = 5, VDIM = 2 };

#define F2(dx, dy, vd) ((dx) + D1D * ((dy) + D1D * (vd)))
#define Q2(vd, c, qx, qy, nc) \
  ((qy) + Q1D * ((qx) + Q1D * ((c) + (nc) * (vd))))
#define F3(dx, dy, dz, vd) \
  ((dx) + D1D * ((dy) + D1D * ((dz) + D1D * (vd))))
#define Q3(vd, c, qx, qy, qz, nc) \
  ((qz) + Q1D * ((qy) + Q1D * ((qx) + Q1D * ((c) + (nc) * (vd)))))

void mfem_interp_value_2d(const double *field, const double *B, double *out) {
  double s0[D1D][Q1D];
  for (int vd = 0; vd < VDIM; ++vd) {
    for (int dy = 0; dy < D1D; ++dy)
      for (int qx = 0; qx < Q1D; ++qx) {
        double acc = 0.0;
        for (int dx = 0; dx < D1D; ++dx)
          acc += B[qx * D1D + dx] * field[F2(dx, dy, vd)];
        s0[dy][qx] = acc;
      }
    for (int qx = 0; qx < Q1D; ++qx)
      for (int qy = 0; qy < Q1D; ++qy) {
        double acc = 0.0;
        for (int dy = 0; dy < D1D; ++dy)
          acc += s0[dy][qx] * B[qy * D1D + dy];
        out[Q2(vd, 0, qx, qy, 1)] = acc;
      }
  }
}

void mfem_interp_grad_2d(const double *field, const double *B,
                         const double *G, double *out) {
  double s0[D1D][Q1D], s1[D1D][Q1D];
  for (int vd = 0; vd < VDIM; ++vd) {
    for (int dy = 0; dy < D1D; ++dy)
      for (int qx = 0; qx < Q1D; ++qx) {
        double u = 0.0, v = 0.0;
        for (int dx = 0; dx < D1D; ++dx) {
          double f = field[F2(dx, dy, vd)];
          u += f * B[qx * D1D + dx];
          v += f * G[qx * D1D + dx];
        }
        s0[dy][qx] = u;
        s1[dy][qx] = v;
      }
    for (int qy = 0; qy < Q1D; ++qy)
      for (int qx = 0; qx < Q1D; ++qx) {
        double u = 0.0, v = 0.0;
        for (int dy = 0; dy < D1D; ++dy) {
          u += s1[dy][qx] * B[qy * D1D + dy];
          v += s0[dy][qx] * G[qy * D1D + dy];
        }
        out[Q2(vd, 0, qx, qy, 2)] = u;
        out[Q2(vd, 1, qx, qy, 2)] = v;
      }
  }
}

void mfem_interp_value_3d(const double *field, const double *B, double *out) {
  double s0[D1D][D1D][Q1D], s1[D1D][Q1D][Q1D];
  for (int vd = 0; vd < VDIM; ++vd) {
    for (int dz = 0; dz < D1D; ++dz)
      for (int dy = 0; dy < D1D; ++dy)
        for (int qx = 0; qx < Q1D; ++qx) {
          double acc = 0.0;
          for (int dx = 0; dx < D1D; ++dx)
            acc += B[qx * D1D + dx] * field[F3(dx, dy, dz, vd)];
          s0[dz][dy][qx] = acc;
        }
    for (int dz = 0; dz < D1D; ++dz)
      for (int qx = 0; qx < Q1D; ++qx)
        for (int qy = 0; qy < Q1D; ++qy) {
          double acc = 0.0;
          for (int dy = 0; dy < D1D; ++dy)
            acc += s0[dz][dy][qx] * B[qy * D1D + dy];
          s1[dz][qy][qx] = acc;
        }
    for (int qz = 0; qz < Q1D; ++qz)
      for (int qy = 0; qy < Q1D; ++qy)
        for (int qx = 0; qx < Q1D; ++qx) {
          double acc = 0.0;
          for (int dz = 0; dz < D1D; ++dz)
            acc += s1[dz][qy][qx] * B[qz * D1D + dz];
          out[Q3(vd, 0, qx, qy, qz, 1)] = acc;
        }
  }
}

void mfem_interp_grad_3d(const double *field, const double *B,
                         const double *G, double *out) {
  double s0[D1D][D1D][Q1D], s1[D1D][D1D][Q1D];
  double s2[D1D][Q1D][Q1D], s3[D1D][Q1D][Q1D], s4[D1D][Q1D][Q1D];
  for (int vd = 0; vd < VDIM; ++vd) {
    for (int dz = 0; dz < D1D; ++dz)
      for (int dy = 0; dy < D1D; ++dy)
        for (int qx = 0; qx < Q1D; ++qx) {
          double u = 0.0, v = 0.0;
          for (int dx = 0; dx < D1D; ++dx) {
            double f = field[F3(dx, dy, dz, vd)];
            u += f * B[qx * D1D + dx];
            v += f * G[qx * D1D + dx];
          }
          s0[dz][dy][qx] = u; s1[dz][dy][qx] = v;
        }
    for (int dz = 0; dz < D1D; ++dz)
      for (int qy = 0; qy < Q1D; ++qy)
        for (int qx = 0; qx < Q1D; ++qx) {
          double u = 0.0, v = 0.0, w = 0.0;
          for (int dy = 0; dy < D1D; ++dy) {
            u += s1[dz][dy][qx] * B[qy * D1D + dy];
            v += s0[dz][dy][qx] * G[qy * D1D + dy];
            w += s0[dz][dy][qx] * B[qy * D1D + dy];
          }
          s2[dz][qy][qx] = u; s3[dz][qy][qx] = v; s4[dz][qy][qx] = w;
        }
    for (int qz = 0; qz < Q1D; ++qz)
      for (int qy = 0; qy < Q1D; ++qy)
        for (int qx = 0; qx < Q1D; ++qx) {
          double u = 0.0, v = 0.0, w = 0.0;
          for (int dz = 0; dz < D1D; ++dz) {
            u += s2[dz][qy][qx] * B[qz * D1D + dz];
            v += s3[dz][qy][qx] * B[qz * D1D + dz];
            w += s4[dz][qy][qx] * G[qz * D1D + dz];
          }
          out[Q3(vd, 0, qx, qy, qz, 3)] = u;
          out[Q3(vd, 1, qx, qy, qz, 3)] = v;
          out[Q3(vd, 2, qx, qy, qz, 3)] = w;
        }
  }
}

void mfem_integrate_value_2d(const double *fqp, const double *B, double *y) {
  double s0[Q1D][D1D];
  for (int vd = 0; vd < VDIM; ++vd) {
    for (int qy = 0; qy < Q1D; ++qy)
      for (int dx = 0; dx < D1D; ++dx) {
        double acc = 0.0;
        for (int qx = 0; qx < Q1D; ++qx)
          acc += fqp[Q2(vd, 0, qx, qy, 1)] * B[qx * D1D + dx];
        s0[qy][dx] = acc;
      }
    for (int dy = 0; dy < D1D; ++dy)
      for (int dx = 0; dx < D1D; ++dx) {
        double acc = 0.0;
        for (int qy = 0; qy < Q1D; ++qy)
          acc += s0[qy][dx] * B[qy * D1D + dy];
        y[F2(dx, dy, vd)] += acc;
      }
  }
}

void mfem_integrate_grad_2d(const double *fqp, const double *B,
                            const double *G, double *y) {
  double s0[Q1D][D1D], s1[Q1D][D1D];
  for (int vd = 0; vd < VDIM; ++vd) {
    for (int qy = 0; qy < Q1D; ++qy)
      for (int dx = 0; dx < D1D; ++dx) {
        double u = 0.0, v = 0.0;
        for (int qx = 0; qx < Q1D; ++qx) {
          u += fqp[Q2(vd, 0, qx, qy, 2)] * G[qx * D1D + dx];
          v += fqp[Q2(vd, 1, qx, qy, 2)] * B[qx * D1D + dx];
        }
        s0[qy][dx] = u; s1[qy][dx] = v;
      }
    for (int dy = 0; dy < D1D; ++dy)
      for (int dx = 0; dx < D1D; ++dx) {
        double u = 0.0, v = 0.0;
        for (int qy = 0; qy < Q1D; ++qy) {
          u += s0[qy][dx] * B[qy * D1D + dy];
          v += s1[qy][dx] * G[qy * D1D + dy];
        }
        y[F2(dx, dy, vd)] += u + v;
      }
  }
}

void mfem_integrate_value_3d(const double *fqp, const double *B, double *y) {
  double s0[Q1D][Q1D][D1D], s1[Q1D][D1D][D1D];
  for (int vd = 0; vd < VDIM; ++vd) {
    for (int qy = 0; qy < Q1D; ++qy)
      for (int dx = 0; dx < D1D; ++dx)
        for (int qz = 0; qz < Q1D; ++qz) {
          double acc = 0.0;
          for (int qx = 0; qx < Q1D; ++qx)
            acc += fqp[Q3(vd, 0, qx, qy, qz, 1)] * B[qx * D1D + dx];
          s0[qz][qy][dx] = acc;
        }
    for (int dy = 0; dy < D1D; ++dy)
      for (int dx = 0; dx < D1D; ++dx)
        for (int qz = 0; qz < Q1D; ++qz) {
          double acc = 0.0;
          for (int qy = 0; qy < Q1D; ++qy)
            acc += s0[qz][qy][dx] * B[qy * D1D + dy];
          s1[qz][dy][dx] = acc;
        }
    for (int dz = 0; dz < D1D; ++dz)
      for (int dy = 0; dy < D1D; ++dy)
        for (int dx = 0; dx < D1D; ++dx) {
          double acc = 0.0;
          for (int qz = 0; qz < Q1D; ++qz)
            acc += s1[qz][dy][dx] * B[qz * D1D + dz];
          y[F3(dx, dy, dz, vd)] += acc;
        }
  }
}

void mfem_integrate_grad_3d(const double *fqp, const double *B,
                            const double *G, double *y) {
  double s0[Q1D][Q1D][D1D], s1[Q1D][Q1D][D1D], s2[Q1D][Q1D][D1D];
  double s3[Q1D][D1D][D1D], s4[Q1D][D1D][D1D], s5[Q1D][D1D][D1D];
  for (int vd = 0; vd < VDIM; ++vd) {
    for (int qz = 0; qz < Q1D; ++qz)
      for (int qy = 0; qy < Q1D; ++qy)
        for (int dx = 0; dx < D1D; ++dx) {
          double u = 0.0, v = 0.0, w = 0.0;
          for (int qx = 0; qx < Q1D; ++qx) {
            u += fqp[Q3(vd, 0, qx, qy, qz, 3)] * G[qx * D1D + dx];
            v += fqp[Q3(vd, 1, qx, qy, qz, 3)] * B[qx * D1D + dx];
            w += fqp[Q3(vd, 2, qx, qy, qz, 3)] * B[qx * D1D + dx];
          }
          s0[qz][qy][dx] = u; s1[qz][qy][dx] = v; s2[qz][qy][dx] = w;
        }
    for (int qz = 0; qz < Q1D; ++qz)
      for (int dy = 0; dy < D1D; ++dy)
        for (int dx = 0; dx < D1D; ++dx) {
          double u = 0.0, v = 0.0, w = 0.0;
          for (int qy = 0; qy < Q1D; ++qy) {
            u += s0[qz][qy][dx] * B[qy * D1D + dy];
            v += s1[qz][qy][dx] * G[qy * D1D + dy];
            w += s2[qz][qy][dx] * B[qy * D1D + dy];
          }
          s3[qz][dy][dx] = u; s4[qz][dy][dx] = v; s5[qz][dy][dx] = w;
        }
    for (int dz = 0; dz < D1D; ++dz)
      for (int dy = 0; dy < D1D; ++dy)
        for (int dx = 0; dx < D1D; ++dx) {
          double u = 0.0, v = 0.0, w = 0.0;
          for (int qz = 0; qz < Q1D; ++qz) {
            u += s3[qz][dy][dx] * B[qz * D1D + dz];
            v += s4[qz][dy][dx] * B[qz * D1D + dz];
            w += s5[qz][dy][dx] * G[qz * D1D + dz];
          }
          y[F3(dx, dy, dz, vd)] += u + v + w;
        }
  }
}
