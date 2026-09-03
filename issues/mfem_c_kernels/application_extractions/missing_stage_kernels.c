/*
 * Concrete FP64 D1D=4, Q1D=5 application kernels omitted from the
 * original MFEM application extraction.  The loop equations follow the PA
 * apply kernels in fem/integ.  They use direct tensor contractions here so
 * that the extracted C records semantics without MFEM's device wrappers.
 */
#ifndef MFEM_BENCH_NE
#define MFEM_BENCH_NE 2
#endif
enum { MF_D=4, MF_E=3, MF_Q=5, MF_NE=MFEM_BENCH_NE,
       MF_Q3=125, MF_D3=64 };
#define MF_S(x,y,z,e) ((x)+MF_D*((y)+MF_D*((z)+MF_D*(e))))
#define MF_V(x,y,z,c,e) ((x)+MF_D*((y)+MF_D*((z)+MF_D*((c)+3*(e)))))
#define MF_QI(x,y,z) ((x)+MF_Q*((y)+MF_Q*(z)))
#define MF_VQ(x,y,z,c,e) (MF_QI(x,y,z)+MF_Q3*((c)+3*(e)))

/* fem/integ/bilininteg_hcurl_kernels.cpp:280-469. */
void mfem_pa_hcurl_mass_apply_3d_direct(
    const double *Bo, const double *Bc, const double *Bot, const double *Bct,
    const double *op, const double *X, double *Y) {
  double q[MF_NE][3][MF_Q][MF_Q][MF_Q];
#define HC_FWD(C,NX,NY,NZ,MX,MY,MZ,OFF) \
  for(int e=0;e<MF_NE;++e)for(int qz=0;qz<MF_Q;++qz) \
  for(int qy=0;qy<MF_Q;++qy)for(int qx=0;qx<MF_Q;++qx){double a=0.; \
  for(int dz=0;dz<NZ;++dz)for(int dy=0;dy<NY;++dy)for(int dx=0;dx<NX;++dx) \
  a+=X[(OFF)+dx+(NX)*(dy+(NY)*dz)+144*e]*(MX)[qx*(NX)+dx]* \
     (MY)[qy*(NY)+dy]*(MZ)[qz*(NZ)+dz];q[e][C][qz][qy][qx]=a;}
  HC_FWD(0,3,4,4,Bo,Bc,Bc,0)
  HC_FWD(1,4,3,4,Bc,Bo,Bc,48)
  HC_FWD(2,4,4,3,Bc,Bc,Bo,96)
#undef HC_FWD
  for(int e=0;e<MF_NE;++e)for(int qz=0;qz<MF_Q;++qz)
  for(int qy=0;qy<MF_Q;++qy)for(int qx=0;qx<MF_Q;++qx){
    int p=MF_QI(qx,qy,qz), o=p+MF_Q3*6*e;
    double x=q[e][0][qz][qy][qx], y=q[e][1][qz][qy][qx], z=q[e][2][qz][qy][qx];
    q[e][0][qz][qy][qx]=op[o]*x+op[o+MF_Q3]*y+op[o+2*MF_Q3]*z;
    q[e][1][qz][qy][qx]=op[o+MF_Q3]*x+op[o+3*MF_Q3]*y+op[o+4*MF_Q3]*z;
    q[e][2][qz][qy][qx]=op[o+2*MF_Q3]*x+op[o+4*MF_Q3]*y+op[o+5*MF_Q3]*z;
  }
#define HC_REV(C,NX,NY,NZ,MX,MY,MZ,OFF) \
  for(int e=0;e<MF_NE;++e)for(int dz=0;dz<NZ;++dz)for(int dy=0;dy<NY;++dy) \
  for(int dx=0;dx<NX;++dx){double a=0.;for(int qz=0;qz<MF_Q;++qz) \
  for(int qy=0;qy<MF_Q;++qy)for(int qx=0;qx<MF_Q;++qx) \
  a+=q[e][C][qz][qy][qx]*(MX)[dx*MF_Q+qx]*(MY)[dy*MF_Q+qy]* \
     (MZ)[dz*MF_Q+qz];Y[(OFF)+dx+(NX)*(dy+(NY)*dz)+144*e]+=a;}
  HC_REV(0,3,4,4,Bot,Bct,Bct,0)
  HC_REV(1,4,3,4,Bct,Bot,Bct,48)
  HC_REV(2,4,4,3,Bct,Bct,Bot,96)
#undef HC_REV
}

/* fem/integ/bilininteg_hdiv_kernels.cpp:471-674. */
void mfem_pa_hdiv_mass_apply_3d_direct(
    const double *Bo, const double *Bc, const double *Bot, const double *Bct,
    const double *op, const double *X, double *Y) {
  double q[MF_NE][3][MF_Q][MF_Q][MF_Q];
#define HD_FWD(C,NX,NY,NZ,MX,MY,MZ,OFF) \
  for(int e=0;e<MF_NE;++e)for(int qz=0;qz<MF_Q;++qz) \
  for(int qy=0;qy<MF_Q;++qy)for(int qx=0;qx<MF_Q;++qx){double a=0.; \
  for(int dz=0;dz<NZ;++dz)for(int dy=0;dy<NY;++dy)for(int dx=0;dx<NX;++dx) \
  a+=X[(OFF)+dx+(NX)*(dy+(NY)*dz)+108*e]*(MX)[qx*(NX)+dx]* \
     (MY)[qy*(NY)+dy]*(MZ)[qz*(NZ)+dz];q[e][C][qz][qy][qx]=a;}
  HD_FWD(0,4,3,3,Bc,Bo,Bo,0)
  HD_FWD(1,3,4,3,Bo,Bc,Bo,36)
  HD_FWD(2,3,3,4,Bo,Bo,Bc,72)
#undef HD_FWD
  for(int e=0;e<MF_NE;++e)for(int qz=0;qz<MF_Q;++qz)
  for(int qy=0;qy<MF_Q;++qy)for(int qx=0;qx<MF_Q;++qx){
    int p=MF_QI(qx,qy,qz), o=p+MF_Q3*6*e;
    double x=q[e][0][qz][qy][qx], y=q[e][1][qz][qy][qx], z=q[e][2][qz][qy][qx];
    q[e][0][qz][qy][qx]=op[o]*x+op[o+MF_Q3]*y+op[o+2*MF_Q3]*z;
    q[e][1][qz][qy][qx]=op[o+MF_Q3]*x+op[o+3*MF_Q3]*y+op[o+4*MF_Q3]*z;
    q[e][2][qz][qy][qx]=op[o+2*MF_Q3]*x+op[o+4*MF_Q3]*y+op[o+5*MF_Q3]*z;
  }
#define HD_REV(C,NX,NY,NZ,MX,MY,MZ,OFF) \
  for(int e=0;e<MF_NE;++e)for(int dz=0;dz<NZ;++dz)for(int dy=0;dy<NY;++dy) \
  for(int dx=0;dx<NX;++dx){double a=0.;for(int qz=0;qz<MF_Q;++qz) \
  for(int qy=0;qy<MF_Q;++qy)for(int qx=0;qx<MF_Q;++qx) \
  a+=q[e][C][qz][qy][qx]*(MX)[dx*MF_Q+qx]*(MY)[dy*MF_Q+qy]* \
     (MZ)[dz*MF_Q+qz];Y[(OFF)+dx+(NX)*(dy+(NY)*dz)+108*e]+=a;}
  HD_REV(0,4,3,3,Bct,Bot,Bot,0)
  HD_REV(1,3,4,3,Bot,Bct,Bot,36)
  HD_REV(2,3,3,4,Bot,Bot,Bct,72)
#undef HD_REV
}

/* fem/integ/bilininteg_vecmass_pa.hpp:100-178, scalar coefficient case. */
void mfem_pa_vector_mass_apply_3d_direct(const double *B, const double *D,
                                         const double *X, double *Y) {
  double q[MF_NE][3][MF_Q][MF_Q][MF_Q];
  for(int e=0;e<MF_NE;++e)for(int c=0;c<3;++c)for(int qz=0;qz<MF_Q;++qz)
  for(int qy=0;qy<MF_Q;++qy)for(int qx=0;qx<MF_Q;++qx){double a=0.;
    for(int dz=0;dz<MF_D;++dz)for(int dy=0;dy<MF_D;++dy)for(int dx=0;dx<MF_D;++dx)
      a+=X[MF_V(dx,dy,dz,c,e)]*B[qx*MF_D+dx]*B[qy*MF_D+dy]*B[qz*MF_D+dz];
    q[e][c][qz][qy][qx]=a*D[MF_QI(qx,qy,qz)+MF_Q3*e];
  }
  for(int e=0;e<MF_NE;++e)for(int c=0;c<3;++c)for(int dz=0;dz<MF_D;++dz)
  for(int dy=0;dy<MF_D;++dy)for(int dx=0;dx<MF_D;++dx){double a=0.;
    for(int qz=0;qz<MF_Q;++qz)for(int qy=0;qy<MF_Q;++qy)for(int qx=0;qx<MF_Q;++qx)
      a+=q[e][c][qz][qy][qx]*B[qx*MF_D+dx]*B[qy*MF_D+dy]*B[qz*MF_D+dz];
    Y[MF_V(dx,dy,dz,c,e)]+=a;
  }
}

/* Scratch-sliced normalization of the same vector-mass operator.  MFEM stores
 * components inside each element, whereas the scalar tensor-product stage
 * expects two adjacent elements.  Pack one component, reuse the already
 * raisable scalar stage, and scatter it back.  The pack/scatter loops are
 * pointwise and raise independently. */
static inline void mfem_pa_vector_mass_component_3d_sliced(
    int c, const double *B, const double *Bt, const double *D,
    const double *X, double *Y) {
  double packed_x[MF_NE * MF_D * MF_D * MF_D];
  double packed_y[MF_NE * MF_D * MF_D * MF_D];
  for (int e = 0; e < MF_NE; ++e)
    for (int dz = 0; dz < MF_D; ++dz)
      for (int dy = 0; dy < MF_D; ++dy)
        for (int dx = 0; dx < MF_D; ++dx) {
          int packed = dx + MF_D * (dy + MF_D * (dz + MF_D * e));
          int vector = MF_V(dx, dy, dz, c, e);
          packed_x[packed] = X[vector];
          packed_y[packed] = Y[vector];
        }
  mfem_pa_mass_apply_3d_stage_sliced(B, Bt, D, packed_x, packed_y);
  for (int e = 0; e < MF_NE; ++e)
    for (int dz = 0; dz < MF_D; ++dz)
      for (int dy = 0; dy < MF_D; ++dy)
        for (int dx = 0; dx < MF_D; ++dx) {
          int packed = dx + MF_D * (dy + MF_D * (dz + MF_D * e));
          Y[MF_V(dx, dy, dz, c, e)] = packed_y[packed];
        }
}

void mfem_pa_vector_mass_apply_3d_sliced(const double *B, const double *D,
                                         const double *X, double *Y) {
  double Bt[MF_D * MF_Q];
  for (int d = 0; d < MF_D; ++d)
    for (int q = 0; q < MF_Q; ++q)
      Bt[d * MF_Q + q] = B[q * MF_D + d];
  mfem_pa_vector_mass_component_3d_sliced(0, B, Bt, D, X, Y);
  mfem_pa_vector_mass_component_3d_sliced(1, B, Bt, D, X, Y);
  mfem_pa_vector_mass_component_3d_sliced(2, B, Bt, D, X, Y);
}

/* fem/integ/bilininteg_vecdiffusion_pa.hpp:96-168. */
void mfem_pa_vector_diffusion_apply_3d_direct(
    const double *B, const double *G, const double *D,
    const double *X, double *Y) {
  double h[MF_NE][3][3][MF_Q][MF_Q][MF_Q];
  for(int e=0;e<MF_NE;++e)for(int c=0;c<3;++c)for(int j=0;j<3;++j)
  for(int qz=0;qz<MF_Q;++qz)for(int qy=0;qy<MF_Q;++qy)for(int qx=0;qx<MF_Q;++qx){
    double g[3]={0.,0.,0.};
    for(int dz=0;dz<MF_D;++dz)for(int dy=0;dy<MF_D;++dy)for(int dx=0;dx<MF_D;++dx){
      double v=X[MF_V(dx,dy,dz,c,e)];
      g[0]+=v*G[qx*MF_D+dx]*B[qy*MF_D+dy]*B[qz*MF_D+dz];
      g[1]+=v*B[qx*MF_D+dx]*G[qy*MF_D+dy]*B[qz*MF_D+dz];
      g[2]+=v*B[qx*MF_D+dx]*B[qy*MF_D+dy]*G[qz*MF_D+dz];
    }
    int p=MF_QI(qx,qy,qz), o=p+MF_Q3*(6*(c+3*e));
    if(j==0)h[e][c][j][qz][qy][qx]=D[o]*g[0]+D[o+MF_Q3]*g[1]+D[o+2*MF_Q3]*g[2];
    if(j==1)h[e][c][j][qz][qy][qx]=D[o+MF_Q3]*g[0]+D[o+3*MF_Q3]*g[1]+D[o+4*MF_Q3]*g[2];
    if(j==2)h[e][c][j][qz][qy][qx]=D[o+2*MF_Q3]*g[0]+D[o+4*MF_Q3]*g[1]+D[o+5*MF_Q3]*g[2];
  }
  for(int e=0;e<MF_NE;++e)for(int c=0;c<3;++c)for(int dz=0;dz<MF_D;++dz)
  for(int dy=0;dy<MF_D;++dy)for(int dx=0;dx<MF_D;++dx){double a=0.;
    for(int qz=0;qz<MF_Q;++qz)for(int qy=0;qy<MF_Q;++qy)for(int qx=0;qx<MF_Q;++qx){
      a+=h[e][c][0][qz][qy][qx]*G[qx*MF_D+dx]*B[qy*MF_D+dy]*B[qz*MF_D+dz];
      a+=h[e][c][1][qz][qy][qx]*B[qx*MF_D+dx]*G[qy*MF_D+dy]*B[qz*MF_D+dz];
      a+=h[e][c][2][qz][qy][qx]*B[qx*MF_D+dx]*B[qy*MF_D+dy]*G[qz*MF_D+dz];
    } Y[MF_V(dx,dy,dz,c,e)]+=a;
  }
}

/* Component-specialized scratch slicing for vector diffusion.  Besides the
 * vector field, pack the six symmetric coefficient planes for each component
 * into the scalar diffusion stage's [element][plane][quadrature] layout. */
static inline void mfem_pa_vector_diffusion_component_3d_sliced(
    int c, const double *B, const double *G, const double *Bt,
    const double *Gt, const double *D, const double *X, double *Y) {
  double packed_d[MF_NE * 6 * MF_Q3];
  double packed_x[MF_NE * MF_D * MF_D * MF_D];
  double packed_y[MF_NE * MF_D * MF_D * MF_D];
  for (int e = 0; e < MF_NE; ++e)
    for (int k = 0; k < 6; ++k)
      for (int p = 0; p < MF_Q3; ++p)
        packed_d[p + MF_Q3 * (k + 6 * e)] =
            D[p + MF_Q3 * (k + 6 * (c + 3 * e))];
  for (int e = 0; e < MF_NE; ++e)
    for (int dz = 0; dz < MF_D; ++dz)
      for (int dy = 0; dy < MF_D; ++dy)
        for (int dx = 0; dx < MF_D; ++dx) {
          int packed = dx + MF_D * (dy + MF_D * (dz + MF_D * e));
          int vector = MF_V(dx, dy, dz, c, e);
          packed_x[packed] = X[vector];
          packed_y[packed] = Y[vector];
        }
  mfem_pa_diffusion_apply_3d_stage_sliced(
      B, G, Bt, Gt, packed_d, packed_x, packed_y);
  for (int e = 0; e < MF_NE; ++e)
    for (int dz = 0; dz < MF_D; ++dz)
      for (int dy = 0; dy < MF_D; ++dy)
        for (int dx = 0; dx < MF_D; ++dx) {
          int packed = dx + MF_D * (dy + MF_D * (dz + MF_D * e));
          Y[MF_V(dx, dy, dz, c, e)] = packed_y[packed];
        }
}

void mfem_pa_vector_diffusion_apply_3d_sliced(
    const double *B, const double *G, const double *D,
    const double *X, double *Y) {
  double Bt[MF_D * MF_Q], Gt[MF_D * MF_Q];
  for (int d = 0; d < MF_D; ++d)
    for (int q = 0; q < MF_Q; ++q) {
      Bt[d * MF_Q + q] = B[q * MF_D + d];
      Gt[d * MF_Q + q] = G[q * MF_D + d];
    }
  mfem_pa_vector_diffusion_component_3d_sliced(0, B, G, Bt, Gt, D, X, Y);
  mfem_pa_vector_diffusion_component_3d_sliced(1, B, G, Bt, Gt, D, X, Y);
  mfem_pa_vector_diffusion_component_3d_sliced(2, B, G, Bt, Gt, D, X, Y);
}

/* fem/integ/nonlininteg_vecconvection_pa.cpp:569-804. */
void mfem_pa_vector_convection_nl_apply_3d_direct(
    const double *B, const double *G, const double *D,
    const double *X, double *Y) {
  double val[MF_NE][3][MF_Q][MF_Q][MF_Q];
  double grad[MF_NE][3][3][MF_Q][MF_Q][MF_Q];
  double z[MF_NE][3][MF_Q][MF_Q][MF_Q];
  for(int e=0;e<MF_NE;++e)for(int c=0;c<3;++c)for(int qz=0;qz<MF_Q;++qz)
  for(int qy=0;qy<MF_Q;++qy)for(int qx=0;qx<MF_Q;++qx){double v=0.,gx=0.,gy=0.,gz=0.;
    for(int dz=0;dz<MF_D;++dz)for(int dy=0;dy<MF_D;++dy)for(int dx=0;dx<MF_D;++dx){
      double u=X[MF_V(dx,dy,dz,c,e)];
      v+=u*B[qx*MF_D+dx]*B[qy*MF_D+dy]*B[qz*MF_D+dz];
      gx+=u*G[qx*MF_D+dx]*B[qy*MF_D+dy]*B[qz*MF_D+dz];
      gy+=u*B[qx*MF_D+dx]*G[qy*MF_D+dy]*B[qz*MF_D+dz];
      gz+=u*B[qx*MF_D+dx]*B[qy*MF_D+dy]*G[qz*MF_D+dz];
    } val[e][c][qz][qy][qx]=v;grad[e][c][0][qz][qy][qx]=gx;
      grad[e][c][1][qz][qy][qx]=gy;grad[e][c][2][qz][qy][qx]=gz;
  }
  for(int e=0;e<MF_NE;++e)for(int cy=0;cy<3;++cy)for(int qz=0;qz<MF_Q;++qz)
  for(int qy=0;qy<MF_Q;++qy)for(int qx=0;qx<MF_Q;++qx){double a=0.;int p=MF_QI(qx,qy,qz);
    for(int c=0;c<3;++c)for(int j=0;j<3;++j)
      a+=val[e][c][qz][qy][qx]*grad[e][cy][j][qz][qy][qx]*D[p+MF_Q3*(j+3*(c+3*e))];
    z[e][cy][qz][qy][qx]=a;
  }
  for(int e=0;e<MF_NE;++e)for(int c=0;c<3;++c)for(int dz=0;dz<MF_D;++dz)
  for(int dy=0;dy<MF_D;++dy)for(int dx=0;dx<MF_D;++dx){double a=0.;
    for(int qz=0;qz<MF_Q;++qz)for(int qy=0;qy<MF_Q;++qy)for(int qx=0;qx<MF_Q;++qx)
      a+=z[e][c][qz][qy][qx]*B[qx*MF_D+dx]*B[qy*MF_D+dy]*B[qz*MF_D+dz];
    Y[MF_V(dx,dy,dz,c,e)]+=a;
  }
}

/* fem/integ/bilininteg_gradient_pa.cpp:462-639. */
void mfem_pa_discrete_gradient_apply_3d_direct(
    const double *B, const double *G, const double *op,
    const double *X, double *Y) {
  double h[MF_NE][3][MF_Q][MF_Q][MF_Q];
  for(int e=0;e<MF_NE;++e)for(int qz=0;qz<MF_Q;++qz)
  for(int qy=0;qy<MF_Q;++qy)for(int qx=0;qx<MF_Q;++qx){double g[3]={0.,0.,0.};
    for(int dz=0;dz<MF_D;++dz)for(int dy=0;dy<MF_D;++dy)for(int dx=0;dx<MF_D;++dx){
      double u=X[MF_S(dx,dy,dz,e)];
      g[0]+=u*G[qx*MF_D+dx]*B[qy*MF_D+dy]*B[qz*MF_D+dz];
      g[1]+=u*B[qx*MF_D+dx]*G[qy*MF_D+dy]*B[qz*MF_D+dz];
      g[2]+=u*B[qx*MF_D+dx]*B[qy*MF_D+dy]*G[qz*MF_D+dz];
    } int p=MF_QI(qx,qy,qz);
    for(int c=0;c<3;++c){double a=0.;for(int j=0;j<3;++j)a+=g[j]*op[p+MF_Q3*(j+3*(c+3*e))];h[e][c][qz][qy][qx]=a;}
  }
  for(int e=0;e<MF_NE;++e)for(int c=0;c<3;++c)for(int dz=0;dz<MF_D;++dz)
  for(int dy=0;dy<MF_D;++dy)for(int dx=0;dx<MF_D;++dx){double a=0.;
    for(int qz=0;qz<MF_Q;++qz)for(int qy=0;qy<MF_Q;++qy)for(int qx=0;qx<MF_Q;++qx)
      a+=h[e][c][qz][qy][qx]*B[qx*MF_D+dx]*B[qy*MF_D+dy]*B[qz*MF_D+dz];
    Y[MF_V(dx,dy,dz,c,e)]+=a;
  }
}

/* Tensor-product interpolation and transpose interpolation for scalar values. */
static inline void mfem_interp_value_3d_stage_sliced(
    const double *x, const double *B, double *q) {
  double sx[MF_NE][MF_D][MF_D][MF_Q];
  double sxy[MF_NE][MF_D][MF_Q][MF_Q];
  for (int e = 0; e < MF_NE; ++e)
    for (int dz = 0; dz < MF_D; ++dz)
      for (int dy = 0; dy < MF_D; ++dy)
        for (int qx = 0; qx < MF_Q; ++qx) {
          double a = 0.0;
          for (int dx = 0; dx < MF_D; ++dx) {
            int i = dx + MF_D * (dy + MF_D * (dz + MF_D * e));
            a += x[i] * B[qx * MF_D + dx];
          }
          sx[e][dz][dy][qx] = a;
        }
  for (int e = 0; e < MF_NE; ++e)
    for (int dz = 0; dz < MF_D; ++dz)
      for (int qy = 0; qy < MF_Q; ++qy)
        for (int qx = 0; qx < MF_Q; ++qx) {
          double a = 0.0;
          for (int dy = 0; dy < MF_D; ++dy)
            a += sx[e][dz][dy][qx] * B[qy * MF_D + dy];
          sxy[e][dz][qy][qx] = a;
        }
  for (int e = 0; e < MF_NE; ++e)
    for (int qz = 0; qz < MF_Q; ++qz)
      for (int qy = 0; qy < MF_Q; ++qy)
        for (int qx = 0; qx < MF_Q; ++qx) {
          double a = 0.0;
          for (int dz = 0; dz < MF_D; ++dz)
            a += sxy[e][dz][qy][qx] * B[qz * MF_D + dz];
          q[MF_QI(qx, qy, qz) + MF_Q3 * e] = a;
        }
}

static inline void mfem_integrate_value_3d_stage_sliced(
    const double *q, const double *Bt, double *y) {
  double tx[MF_NE][MF_Q][MF_Q][MF_D];
  double txy[MF_NE][MF_Q][MF_D][MF_D];
  double u[MF_NE][MF_D][MF_D][MF_D];
  for (int e = 0; e < MF_NE; ++e)
    for (int qz = 0; qz < MF_Q; ++qz)
      for (int qy = 0; qy < MF_Q; ++qy)
        for (int dx = 0; dx < MF_D; ++dx) {
          double a = 0.0;
          for (int qx = 0; qx < MF_Q; ++qx)
            a += q[MF_QI(qx, qy, qz) + MF_Q3 * e] *
                 Bt[dx * MF_Q + qx];
          tx[e][qz][qy][dx] = a;
        }
  for (int e = 0; e < MF_NE; ++e)
    for (int qz = 0; qz < MF_Q; ++qz)
      for (int dy = 0; dy < MF_D; ++dy)
        for (int dx = 0; dx < MF_D; ++dx) {
          double a = 0.0;
          for (int qy = 0; qy < MF_Q; ++qy)
            a += tx[e][qz][qy][dx] * Bt[dy * MF_Q + qy];
          txy[e][qz][dy][dx] = a;
        }
  for (int e = 0; e < MF_NE; ++e)
    for (int dz = 0; dz < MF_D; ++dz)
      for (int dy = 0; dy < MF_D; ++dy)
        for (int dx = 0; dx < MF_D; ++dx) {
          double a = 0.0;
          for (int qz = 0; qz < MF_Q; ++qz)
            a += txy[e][qz][dy][dx] * Bt[dz * MF_Q + qz];
          u[e][dz][dy][dx] = a;
        }
  for (int e = 0; e < MF_NE; ++e)
    for (int dz = 0; dz < MF_D; ++dz)
      for (int dy = 0; dy < MF_D; ++dy)
        for (int dx = 0; dx < MF_D; ++dx) {
          int i = dx + MF_D * (dy + MF_D * (dz + MF_D * e));
          y[i] += u[e][dz][dy][dx];
        }
}

static inline void mfem_pa_discrete_gradient_component_3d_sliced(
    int c, const double *h, const double *Bt, double *Y) {
  double packed_q[MF_NE * MF_Q3];
  double packed_y[MF_NE * MF_D * MF_D * MF_D];
  for (int e = 0; e < MF_NE; ++e) {
    for (int p = 0; p < MF_Q3; ++p)
      packed_q[p + MF_Q3 * e] = h[p + MF_Q3 * (c + 3 * e)];
    for (int dz = 0; dz < MF_D; ++dz)
      for (int dy = 0; dy < MF_D; ++dy)
        for (int dx = 0; dx < MF_D; ++dx) {
          int packed = dx + MF_D * (dy + MF_D * (dz + MF_D * e));
          packed_y[packed] = Y[MF_V(dx, dy, dz, c, e)];
        }
  }
  mfem_integrate_value_3d_stage_sliced(packed_q, Bt, packed_y);
  for (int e = 0; e < MF_NE; ++e)
    for (int dz = 0; dz < MF_D; ++dz)
      for (int dy = 0; dy < MF_D; ++dy)
        for (int dx = 0; dx < MF_D; ++dx) {
          int packed = dx + MF_D * (dy + MF_D * (dz + MF_D * e));
          Y[MF_V(dx, dy, dz, c, e)] = packed_y[packed];
        }
}

void mfem_pa_discrete_gradient_apply_3d_sliced(
    const double *B, const double *G, const double *op,
    const double *X, double *Y) {
  double Bt[MF_D * MF_Q];
  double grad[MF_NE * 3 * MF_Q3];
  double h[MF_NE * 3 * MF_Q3];
  for (int d = 0; d < MF_D; ++d)
    for (int q = 0; q < MF_Q; ++q)
      Bt[d * MF_Q + q] = B[q * MF_D + d];
  mfem_interp_grad_3d_stage_sliced(X, B, G, grad);
  for (int e = 0; e < MF_NE; ++e)
    for (int c = 0; c < 3; ++c)
      for (int qz = 0; qz < MF_Q; ++qz)
        for (int qy = 0; qy < MF_Q; ++qy)
          for (int qx = 0; qx < MF_Q; ++qx) {
            int p = MF_QI(qx, qy, qz);
            double a = 0.0;
            for (int j = 0; j < 3; ++j) {
              int grad_i = qz + MF_Q *
                  (qy + MF_Q * (qx + MF_Q * (j + 3 * e)));
              a += grad[grad_i] * op[p + MF_Q3 * (j + 3 * (c + 3 * e))];
            }
            h[p + MF_Q3 * (c + 3 * e)] = a;
          }
  mfem_pa_discrete_gradient_component_3d_sliced(0, h, Bt, Y);
  mfem_pa_discrete_gradient_component_3d_sliced(1, h, Bt, Y);
  mfem_pa_discrete_gradient_component_3d_sliced(2, h, Bt, Y);
}

static inline void mfem_pa_vector_convection_interp_component_3d_sliced(
    int c, const double *B, const double *G, const double *X,
    double *value, double *grad) {
  double packed_x[MF_NE * MF_D * MF_D * MF_D];
  double packed_value[MF_NE * MF_Q3];
  double packed_grad[MF_NE * 3 * MF_Q3];
  for (int e = 0; e < MF_NE; ++e)
    for (int dz = 0; dz < MF_D; ++dz)
      for (int dy = 0; dy < MF_D; ++dy)
        for (int dx = 0; dx < MF_D; ++dx) {
          int packed = dx + MF_D * (dy + MF_D * (dz + MF_D * e));
          packed_x[packed] = X[MF_V(dx, dy, dz, c, e)];
        }
  mfem_interp_value_3d_stage_sliced(packed_x, B, packed_value);
  mfem_interp_grad_3d_stage_sliced(packed_x, B, G, packed_grad);
  for (int e = 0; e < MF_NE; ++e)
    for (int qz = 0; qz < MF_Q; ++qz)
      for (int qy = 0; qy < MF_Q; ++qy)
        for (int qx = 0; qx < MF_Q; ++qx) {
          int p = MF_QI(qx, qy, qz);
          value[p + MF_Q3 * (c + 3 * e)] = packed_value[p + MF_Q3 * e];
        }
  /* Keep the value and gradient scatters as separate pointwise stages.  This
   * avoids wrapping two dependent linalg.generics in the same affine loop. */
  for (int e = 0; e < MF_NE; ++e)
    for (int j = 0; j < 3; ++j)
      for (int qz = 0; qz < MF_Q; ++qz)
        for (int qy = 0; qy < MF_Q; ++qy)
          for (int qx = 0; qx < MF_Q; ++qx) {
            int p = MF_QI(qx, qy, qz);
            int packed_g = qz + MF_Q *
                (qy + MF_Q * (qx + MF_Q * (j + 3 * e)));
            grad[p + MF_Q3 * (j + 3 * (c + 3 * e))] = packed_grad[packed_g];
          }
}

void mfem_pa_vector_convection_nl_apply_3d_sliced(
    const double *B, const double *G, const double *D,
    const double *X, double *Y) {
  double Bt[MF_D * MF_Q];
  double value[MF_NE * 3 * MF_Q3];
  double grad[MF_NE * 3 * 3 * MF_Q3];
  double z[MF_NE * 3 * MF_Q3];
  for (int d = 0; d < MF_D; ++d)
    for (int q = 0; q < MF_Q; ++q)
      Bt[d * MF_Q + q] = B[q * MF_D + d];
  mfem_pa_vector_convection_interp_component_3d_sliced(
      0, B, G, X, value, grad);
  mfem_pa_vector_convection_interp_component_3d_sliced(
      1, B, G, X, value, grad);
  mfem_pa_vector_convection_interp_component_3d_sliced(
      2, B, G, X, value, grad);
  for (int e = 0; e < MF_NE; ++e)
    for (int cy = 0; cy < 3; ++cy)
      for (int qz = 0; qz < MF_Q; ++qz)
        for (int qy = 0; qy < MF_Q; ++qy)
          for (int qx = 0; qx < MF_Q; ++qx) {
            int p = MF_QI(qx, qy, qz);
            double a = 0.0;
            for (int c = 0; c < 3; ++c)
              for (int j = 0; j < 3; ++j)
                a += value[p + MF_Q3 * (c + 3 * e)] *
                     grad[p + MF_Q3 * (j + 3 * (cy + 3 * e))] *
                     D[p + MF_Q3 * (j + 3 * (c + 3 * e))];
            z[p + MF_Q3 * (cy + 3 * e)] = a;
          }
  mfem_pa_discrete_gradient_component_3d_sliced(0, z, Bt, Y);
  mfem_pa_discrete_gradient_component_3d_sliced(1, z, Bt, Y);
  mfem_pa_discrete_gradient_component_3d_sliced(2, z, Bt, Y);
}

/* fem/integ/bilininteg_vecdiv_pa.cpp:870-1060. */
void mfem_pa_discrete_divergence_apply_3d_direct(
    const double *B, const double *G, const double *op,
    const double *X, double *Y) {
  double q[MF_NE][MF_Q][MF_Q][MF_Q];
  for(int e=0;e<MF_NE;++e)for(int qz=0;qz<MF_Q;++qz)
  for(int qy=0;qy<MF_Q;++qy)for(int qx=0;qx<MF_Q;++qx){double a=0.;int p=MF_QI(qx,qy,qz);
    for(int c=0;c<3;++c)for(int dz=0;dz<MF_D;++dz)for(int dy=0;dy<MF_D;++dy)
    for(int dx=0;dx<MF_D;++dx){double u=X[MF_V(dx,dy,dz,c,e)];
      a+=u*G[qx*MF_D+dx]*B[qy*MF_D+dy]*B[qz*MF_D+dz]*op[p+MF_Q3*(0+3*(c+3*e))];
      a+=u*B[qx*MF_D+dx]*G[qy*MF_D+dy]*B[qz*MF_D+dz]*op[p+MF_Q3*(1+3*(c+3*e))];
      a+=u*B[qx*MF_D+dx]*B[qy*MF_D+dy]*G[qz*MF_D+dz]*op[p+MF_Q3*(2+3*(c+3*e))];
    } q[e][qz][qy][qx]=a;
  }
  for(int e=0;e<MF_NE;++e)for(int dz=0;dz<MF_D;++dz)for(int dy=0;dy<MF_D;++dy)
  for(int dx=0;dx<MF_D;++dx){double a=0.;for(int qz=0;qz<MF_Q;++qz){
  for(int qy=0;qy<MF_Q;++qy)for(int qx=0;qx<MF_Q;++qx){
    a+=q[e][qz][qy][qx]*B[qx*MF_D+dx]*B[qy*MF_D+dy]*B[qz*MF_D+dz];}
  }
    Y[MF_S(dx,dy,dz,e)]+=a;
  }
}

/* One preconditioned-CG algebra step used by ex9p's mass inversion. */
void mfem_mass_pcg_step_2d(const double *Ap, const double *inv_diag,
    double alpha, double beta, double *x, double *r, double *z, double *p,
    double *r_dot_z) {
  double sum=0.;
  /* Keep vector updates and the scalar dot-product reduction as distinct
   * stages.  Combining them creates one multi-output Linalg reduction whose
   * vector outputs are indexed by the reduction iterator while the scalar
   * output aliases one slot; that representation has no sound DPS lowering. */
  for(int i=0;i<16*MFEM_BENCH_NE;++i) x[i]+=alpha*p[i];
  for(int i=0;i<16*MFEM_BENCH_NE;++i) r[i]-=alpha*Ap[i];
  for(int i=0;i<16*MFEM_BENCH_NE;++i) z[i]=inv_diag[i]*r[i];
  for(int i=0;i<16*MFEM_BENCH_NE;++i) sum+=r[i]*z[i];
  for(int i=0;i<16*MFEM_BENCH_NE;++i)p[i]=z[i]+beta*p[i];
  *r_dot_z=sum;
}

#undef MF_VQ
#undef MF_QI
#undef MF_V
#undef MF_S
