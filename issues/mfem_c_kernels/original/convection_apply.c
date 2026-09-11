/* Concrete C extraction of MFEM PA convection apply kernels. */
enum { D1D = 4, Q1D = 5, NE = 2 };
#define X2(dx,dy,e) ((dx) + D1D*((dy) + D1D*(e)))
#define X3(dx,dy,dz,e) ((dx) + D1D*((dy) + D1D*((dz) + D1D*(e))))
#define OP2(qx,qy,c,e) ((qx) + Q1D*((qy) + Q1D*((c) + 2*(e))))
#define OP3(qx,qy,qz,c,e) \
  ((qx) + Q1D*((qy) + Q1D*((qz) + Q1D*((c) + 3*(e)))))

void mfem_pa_convection_apply_2d(const double *B, const double *G,
                                 const double *Bt, const double *op,
                                 const double *x, double *y) {
  for (int e = 0; e < NE; ++e) {
    double u[D1D][D1D], Bu[D1D][Q1D], Gu[D1D][Q1D];
    double GBu[Q1D][Q1D], BGu[Q1D][Q1D], DGu[Q1D][Q1D];
    double BDGu[D1D][Q1D];
    for (int dy = 0; dy < D1D; ++dy)
      for (int dx = 0; dx < D1D; ++dx) u[dy][dx] = x[X2(dx,dy,e)];
    for (int dy = 0; dy < D1D; ++dy)
      for (int qx = 0; qx < Q1D; ++qx) {
        Bu[dy][qx] = Gu[dy][qx] = 0.0;
        for (int dx = 0; dx < D1D; ++dx) {
          Bu[dy][qx] += B[qx*D1D+dx] * u[dy][dx];
          Gu[dy][qx] += G[qx*D1D+dx] * u[dy][dx];
        }
      }
    for (int qx = 0; qx < Q1D; ++qx)
      for (int qy = 0; qy < Q1D; ++qy) {
        GBu[qy][qx] = BGu[qy][qx] = 0.0;
        for (int dy = 0; dy < D1D; ++dy) {
          GBu[qy][qx] += G[qy*D1D+dy] * Bu[dy][qx];
          BGu[qy][qx] += B[qy*D1D+dy] * Gu[dy][qx];
        }
        DGu[qy][qx] = op[OP2(qx,qy,0,e)] * BGu[qy][qx]
                    + op[OP2(qx,qy,1,e)] * GBu[qy][qx];
      }
    for (int qx = 0; qx < Q1D; ++qx)
      for (int dy = 0; dy < D1D; ++dy) {
        BDGu[dy][qx] = 0.0;
        for (int qy = 0; qy < Q1D; ++qy)
          BDGu[dy][qx] += Bt[dy*Q1D+qy] * DGu[qy][qx];
      }
    for (int dx = 0; dx < D1D; ++dx)
      for (int dy = 0; dy < D1D; ++dy)
        for (int qx = 0; qx < Q1D; ++qx)
          y[X2(dx,dy,e)] += Bt[dx*Q1D+qx] * BDGu[dy][qx];
  }
}

void mfem_pa_convection_apply_3d(const double *B, const double *G,
                                 const double *Bt, const double *op,
                                 const double *x, double *y) {
  for (int e = 0; e < NE; ++e) {
    double u[D1D][D1D][D1D];
    double Bu[D1D][D1D][Q1D], Gu[D1D][D1D][Q1D];
    double BBu[D1D][Q1D][Q1D], GBu[D1D][Q1D][Q1D], BGu[D1D][Q1D][Q1D];
    double GBBu[Q1D][Q1D][Q1D], BGBu[Q1D][Q1D][Q1D];
    double BBGu[Q1D][Q1D][Q1D], DGu[Q1D][Q1D][Q1D];
    double BDGu[D1D][Q1D][Q1D], BBDGu[D1D][D1D][Q1D];
    for (int dz=0; dz<D1D; ++dz) for (int dy=0; dy<D1D; ++dy)
      for (int dx=0; dx<D1D; ++dx) u[dz][dy][dx]=x[X3(dx,dy,dz,e)];
    for (int dz=0; dz<D1D; ++dz) for (int dy=0; dy<D1D; ++dy)
      for (int qx=0; qx<Q1D; ++qx) {
        Bu[dz][dy][qx]=Gu[dz][dy][qx]=0.0;
        for (int dx=0; dx<D1D; ++dx) {
          Bu[dz][dy][qx] += B[qx*D1D+dx]*u[dz][dy][dx];
          Gu[dz][dy][qx] += G[qx*D1D+dx]*u[dz][dy][dx];
        }
      }
    for (int dz=0; dz<D1D; ++dz) for (int qx=0; qx<Q1D; ++qx)
      for (int qy=0; qy<Q1D; ++qy) {
        BBu[dz][qy][qx]=GBu[dz][qy][qx]=BGu[dz][qy][qx]=0.0;
        for (int dy=0; dy<D1D; ++dy) {
          BBu[dz][qy][qx] += B[qy*D1D+dy]*Bu[dz][dy][qx];
          GBu[dz][qy][qx] += G[qy*D1D+dy]*Bu[dz][dy][qx];
          BGu[dz][qy][qx] += B[qy*D1D+dy]*Gu[dz][dy][qx];
        }
      }
    for (int qx=0; qx<Q1D; ++qx) for (int qy=0; qy<Q1D; ++qy)
      for (int qz=0; qz<Q1D; ++qz) {
        GBBu[qz][qy][qx]=BGBu[qz][qy][qx]=BBGu[qz][qy][qx]=0.0;
        for (int dz=0; dz<D1D; ++dz) {
          GBBu[qz][qy][qx] += G[qz*D1D+dz]*BBu[dz][qy][qx];
          BGBu[qz][qy][qx] += B[qz*D1D+dz]*GBu[dz][qy][qx];
          BBGu[qz][qy][qx] += B[qz*D1D+dz]*BGu[dz][qy][qx];
        }
        DGu[qz][qy][qx] = op[OP3(qx,qy,qz,0,e)]*BBGu[qz][qy][qx]
                          + op[OP3(qx,qy,qz,1,e)]*BGBu[qz][qy][qx]
                          + op[OP3(qx,qy,qz,2,e)]*GBBu[qz][qy][qx];
      }
    for (int qx=0; qx<Q1D; ++qx) for (int qy=0; qy<Q1D; ++qy)
      for (int dz=0; dz<D1D; ++dz) {
        BDGu[dz][qy][qx]=0.0;
        for (int qz=0; qz<Q1D; ++qz)
          BDGu[dz][qy][qx] += Bt[dz*Q1D+qz]*DGu[qz][qy][qx];
      }
    for (int dz=0; dz<D1D; ++dz) for (int qx=0; qx<Q1D; ++qx)
      for (int dy=0; dy<D1D; ++dy) {
        BBDGu[dz][dy][qx]=0.0;
        for (int qy=0; qy<Q1D; ++qy)
          BBDGu[dz][dy][qx] += Bt[dy*Q1D+qy]*BDGu[dz][qy][qx];
      }
    for (int dz=0; dz<D1D; ++dz) for (int dy=0; dy<D1D; ++dy)
      for (int dx=0; dx<D1D; ++dx) for (int qx=0; qx<Q1D; ++qx)
        y[X3(dx,dy,dz,e)] += Bt[dx*Q1D+qx]*BBDGu[dz][dy][qx];
  }
}
