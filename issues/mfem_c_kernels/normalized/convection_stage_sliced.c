/* PA convection with disjoint element/channel/stage scratch. */
#ifndef MFEM_BENCH_NE
#define MFEM_BENCH_NE 2
#endif
enum { D1D=4,Q1D=5,NE=MFEM_BENCH_NE };
#define X2(x,y,e) ((x)+D1D*((y)+D1D*(e)))
#define X3(x,y,z,e) ((x)+D1D*((y)+D1D*((z)+D1D*(e))))
#define O2(x,y,c,e) ((x)+Q1D*((y)+Q1D*((c)+2*(e))))
#define O3(x,y,z,c,e) ((x)+Q1D*((y)+Q1D*((z)+Q1D*((c)+3*(e)))))

// polygeist-arg-extents mfem_pa_convection_apply_2d_stage_sliced: B=20, G=20, Bt=20, op=50*MFEM_BENCH_NE, X=16*MFEM_BENCH_NE, Y=16*MFEM_BENCH_NE
void mfem_pa_convection_apply_2d_stage_sliced(const double *B,const double *G,
 const double *Bt,const double *op,const double *X,double *Y) {
  double bx[NE][D1D][Q1D],gx[NE][D1D][Q1D],g0[NE][Q1D][Q1D],g1[NE][Q1D][Q1D];
  double h[NE][Q1D][Q1D],tx[NE][Q1D][D1D],u[NE][D1D][D1D];
  for(int e=0;e<NE;++e)for(int dy=0;dy<D1D;++dy)for(int qx=0;qx<Q1D;++qx){double a=0.;for(int dx=0;dx<D1D;++dx)a+=X[X2(dx,dy,e)]*B[qx*D1D+dx];bx[e][dy][qx]=a;}
  for(int e=0;e<NE;++e)for(int dy=0;dy<D1D;++dy)for(int qx=0;qx<Q1D;++qx){double a=0.;for(int dx=0;dx<D1D;++dx)a+=X[X2(dx,dy,e)]*G[qx*D1D+dx];gx[e][dy][qx]=a;}
  for(int e=0;e<NE;++e)for(int qy=0;qy<Q1D;++qy)for(int qx=0;qx<Q1D;++qx){double a=0.;for(int dy=0;dy<D1D;++dy)a+=gx[e][dy][qx]*B[qy*D1D+dy];g0[e][qy][qx]=a;}
  for(int e=0;e<NE;++e)for(int qy=0;qy<Q1D;++qy)for(int qx=0;qx<Q1D;++qx){double a=0.;for(int dy=0;dy<D1D;++dy)a+=bx[e][dy][qx]*G[qy*D1D+dy];g1[e][qy][qx]=a;}
  for(int e=0;e<NE;++e)for(int qy=0;qy<Q1D;++qy)for(int qx=0;qx<Q1D;++qx)h[e][qy][qx]=op[O2(qx,qy,0,e)]*g0[e][qy][qx]+op[O2(qx,qy,1,e)]*g1[e][qy][qx];
  for(int e=0;e<NE;++e)for(int qy=0;qy<Q1D;++qy)for(int dx=0;dx<D1D;++dx){double a=0.;for(int qx=0;qx<Q1D;++qx)a+=h[e][qy][qx]*Bt[dx*Q1D+qx];tx[e][qy][dx]=a;}
  for(int e=0;e<NE;++e)for(int dy=0;dy<D1D;++dy)for(int dx=0;dx<D1D;++dx){double a=0.;for(int qy=0;qy<Q1D;++qy)a+=tx[e][qy][dx]*Bt[dy*Q1D+qy];u[e][dy][dx]=a;}
  for(int e=0;e<NE;++e)for(int dy=0;dy<D1D;++dy)for(int dx=0;dx<D1D;++dx)Y[X2(dx,dy,e)]+=u[e][dy][dx];
}

// polygeist-arg-extents mfem_pa_convection_apply_3d_stage_sliced: B=20, G=20, Bt=20, op=375*MFEM_BENCH_NE, X=64*MFEM_BENCH_NE, Y=64*MFEM_BENCH_NE
void mfem_pa_convection_apply_3d_stage_sliced(const double *B,const double *G,
 const double *Bt,const double *op,const double *X,double *Y) {
  double bx[NE][D1D][D1D][Q1D],gx[NE][D1D][D1D][Q1D];
  double ix[NE][D1D][Q1D][Q1D],iy[NE][D1D][Q1D][Q1D],iz[NE][D1D][Q1D][Q1D];
  double g0[NE][Q1D][Q1D][Q1D],g1[NE][Q1D][Q1D][Q1D],g2[NE][Q1D][Q1D][Q1D];
  double h[NE][Q1D][Q1D][Q1D],tx[NE][Q1D][Q1D][D1D],txy[NE][Q1D][D1D][D1D],u[NE][D1D][D1D][D1D];
#define XS(dst,M) for(int e=0;e<NE;++e)for(int dz=0;dz<D1D;++dz)for(int dy=0;dy<D1D;++dy)for(int qx=0;qx<Q1D;++qx){double a=0.;for(int dx=0;dx<D1D;++dx)a+=X[X3(dx,dy,dz,e)]*M[qx*D1D+dx];dst[e][dz][dy][qx]=a;}
  XS(bx,B) XS(gx,G)
#undef XS
#define YS(dst,SRC,M) for(int e=0;e<NE;++e)for(int dz=0;dz<D1D;++dz)for(int qy=0;qy<Q1D;++qy)for(int qx=0;qx<Q1D;++qx){double a=0.;for(int dy=0;dy<D1D;++dy)a+=SRC[e][dz][dy][qx]*M[qy*D1D+dy];dst[e][dz][qy][qx]=a;}
  YS(ix,gx,B) YS(iy,bx,G) YS(iz,bx,B)
#undef YS
#define ZS(dst,SRC,M) for(int e=0;e<NE;++e)for(int qz=0;qz<Q1D;++qz)for(int qy=0;qy<Q1D;++qy)for(int qx=0;qx<Q1D;++qx){double a=0.;for(int dz=0;dz<D1D;++dz)a+=SRC[e][dz][qy][qx]*M[qz*D1D+dz];dst[e][qz][qy][qx]=a;}
  ZS(g0,ix,B) ZS(g1,iy,B) ZS(g2,iz,G)
#undef ZS
  for(int e=0;e<NE;++e)for(int qz=0;qz<Q1D;++qz)for(int qy=0;qy<Q1D;++qy)for(int qx=0;qx<Q1D;++qx)h[e][qz][qy][qx]=op[O3(qx,qy,qz,0,e)]*g0[e][qz][qy][qx]+op[O3(qx,qy,qz,1,e)]*g1[e][qz][qy][qx]+op[O3(qx,qy,qz,2,e)]*g2[e][qz][qy][qx];
  for(int e=0;e<NE;++e)for(int qz=0;qz<Q1D;++qz)for(int qy=0;qy<Q1D;++qy)for(int dx=0;dx<D1D;++dx){double a=0.;for(int qx=0;qx<Q1D;++qx)a+=h[e][qz][qy][qx]*Bt[dx*Q1D+qx];tx[e][qz][qy][dx]=a;}
  for(int e=0;e<NE;++e)for(int qz=0;qz<Q1D;++qz)for(int dy=0;dy<D1D;++dy)for(int dx=0;dx<D1D;++dx){double a=0.;for(int qy=0;qy<Q1D;++qy)a+=tx[e][qz][qy][dx]*Bt[dy*Q1D+qy];txy[e][qz][dy][dx]=a;}
  for(int e=0;e<NE;++e)for(int dz=0;dz<D1D;++dz)for(int dy=0;dy<D1D;++dy)for(int dx=0;dx<D1D;++dx){double a=0.;for(int qz=0;qz<Q1D;++qz)a+=txy[e][qz][dy][dx]*Bt[dz*Q1D+qz];u[e][dz][dy][dx]=a;}
  for(int e=0;e<NE;++e)for(int dz=0;dz<D1D;++dz)for(int dy=0;dy<D1D;++dy)for(int dx=0;dx<D1D;++dx)Y[X3(dx,dy,dz,e)]+=u[e][dz][dy][dx];
}
