/* Symmetric PA diffusion with disjoint element/channel/stage scratch. */
enum { D1D=4,Q1D=5,NE=2 };
#define X2(x,y,e) ((x)+D1D*((y)+D1D*(e)))
#define X3(x,y,z,e) ((x)+D1D*((y)+D1D*((z)+D1D*(e))))
#define O2(q,c,e) ((q)+Q1D*Q1D*((c)+3*(e)))
#define O3(q,c,e) ((q)+Q1D*Q1D*Q1D*((c)+6*(e)))

void mfem_pa_diffusion_apply_2d_stage_sliced(const double *B,const double *G,
 const double *Bt,const double *Gt,const double *D,const double *X,double *Y) {
  double bx[NE][D1D][Q1D],gx[NE][D1D][Q1D],g0[NE][Q1D][Q1D],g1[NE][Q1D][Q1D];
  double h0[NE][Q1D][Q1D],h1[NE][Q1D][Q1D],tx0[NE][Q1D][D1D],tx1[NE][Q1D][D1D];
  double u0[NE][D1D][D1D],u1[NE][D1D][D1D];
  for(int e=0;e<NE;++e) for(int dy=0;dy<D1D;++dy) for(int qx=0;qx<Q1D;++qx){double a=0.;for(int dx=0;dx<D1D;++dx)a+=X[X2(dx,dy,e)]*B[qx*D1D+dx];bx[e][dy][qx]=a;}
  for(int e=0;e<NE;++e) for(int dy=0;dy<D1D;++dy) for(int qx=0;qx<Q1D;++qx){double a=0.;for(int dx=0;dx<D1D;++dx)a+=X[X2(dx,dy,e)]*G[qx*D1D+dx];gx[e][dy][qx]=a;}
  for(int e=0;e<NE;++e) for(int qy=0;qy<Q1D;++qy) for(int qx=0;qx<Q1D;++qx){double a=0.;for(int dy=0;dy<D1D;++dy)a+=gx[e][dy][qx]*B[qy*D1D+dy];g0[e][qy][qx]=a;}
  for(int e=0;e<NE;++e) for(int qy=0;qy<Q1D;++qy) for(int qx=0;qx<Q1D;++qx){double a=0.;for(int dy=0;dy<D1D;++dy)a+=bx[e][dy][qx]*G[qy*D1D+dy];g1[e][qy][qx]=a;}
  for(int e=0;e<NE;++e) for(int qy=0;qy<Q1D;++qy) for(int qx=0;qx<Q1D;++qx){int q=qx+Q1D*qy;h0[e][qy][qx]=D[O2(q,0,e)]*g0[e][qy][qx]+D[O2(q,1,e)]*g1[e][qy][qx];h1[e][qy][qx]=D[O2(q,1,e)]*g0[e][qy][qx]+D[O2(q,2,e)]*g1[e][qy][qx];}
  for(int e=0;e<NE;++e) for(int qy=0;qy<Q1D;++qy) for(int dx=0;dx<D1D;++dx){double a=0.;for(int qx=0;qx<Q1D;++qx)a+=h0[e][qy][qx]*Gt[dx*Q1D+qx];tx0[e][qy][dx]=a;}
  for(int e=0;e<NE;++e) for(int qy=0;qy<Q1D;++qy) for(int dx=0;dx<D1D;++dx){double a=0.;for(int qx=0;qx<Q1D;++qx)a+=h1[e][qy][qx]*Bt[dx*Q1D+qx];tx1[e][qy][dx]=a;}
  for(int e=0;e<NE;++e) for(int dy=0;dy<D1D;++dy) for(int dx=0;dx<D1D;++dx){double a=0.;for(int qy=0;qy<Q1D;++qy)a+=tx0[e][qy][dx]*Bt[dy*Q1D+qy];u0[e][dy][dx]=a;}
  for(int e=0;e<NE;++e) for(int dy=0;dy<D1D;++dy) for(int dx=0;dx<D1D;++dx){double a=0.;for(int qy=0;qy<Q1D;++qy)a+=tx1[e][qy][dx]*Gt[dy*Q1D+qy];u1[e][dy][dx]=a;}
  for(int e=0;e<NE;++e) for(int dy=0;dy<D1D;++dy) for(int dx=0;dx<D1D;++dx)Y[X2(dx,dy,e)]+=u0[e][dy][dx]+u1[e][dy][dx];
}

void mfem_pa_diffusion_apply_3d_stage_sliced(const double *B,const double *G,
 const double *Bt,const double *Gt,const double *D,const double *X,double *Y) {
  double bx[NE][D1D][D1D][Q1D],gx[NE][D1D][D1D][Q1D];
  double ix[NE][D1D][Q1D][Q1D],iy[NE][D1D][Q1D][Q1D],iz[NE][D1D][Q1D][Q1D];
  double g0[NE][Q1D][Q1D][Q1D],g1[NE][Q1D][Q1D][Q1D],g2[NE][Q1D][Q1D][Q1D];
  double h0[NE][Q1D][Q1D][Q1D],h1[NE][Q1D][Q1D][Q1D],h2[NE][Q1D][Q1D][Q1D];
  double rx0[NE][Q1D][Q1D][D1D],rx1[NE][Q1D][Q1D][D1D],rx2[NE][Q1D][Q1D][D1D];
  double ry0[NE][Q1D][D1D][D1D],ry1[NE][Q1D][D1D][D1D],ry2[NE][Q1D][D1D][D1D];
  double u0[NE][D1D][D1D][D1D],u1[NE][D1D][D1D][D1D],u2[NE][D1D][D1D][D1D];
#define RED4(dst,EXPR) for(int e=0;e<NE;++e)for(int dz=0;dz<D1D;++dz)for(int dy=0;dy<D1D;++dy)for(int qx=0;qx<Q1D;++qx){double a=0.;for(int dx=0;dx<D1D;++dx)a+=(EXPR);dst[e][dz][dy][qx]=a;}
  RED4(bx,X[X3(dx,dy,dz,e)]*B[qx*D1D+dx]) RED4(gx,X[X3(dx,dy,dz,e)]*G[qx*D1D+dx])
#undef RED4
#define YST(dst,SRC,MAT) for(int e=0;e<NE;++e)for(int dz=0;dz<D1D;++dz)for(int qy=0;qy<Q1D;++qy)for(int qx=0;qx<Q1D;++qx){double a=0.;for(int dy=0;dy<D1D;++dy)a+=SRC[e][dz][dy][qx]*MAT[qy*D1D+dy];dst[e][dz][qy][qx]=a;}
  YST(ix,gx,B) YST(iy,bx,G) YST(iz,bx,B)
#undef YST
#define ZST(dst,SRC,MAT) for(int e=0;e<NE;++e)for(int qz=0;qz<Q1D;++qz)for(int qy=0;qy<Q1D;++qy)for(int qx=0;qx<Q1D;++qx){double a=0.;for(int dz=0;dz<D1D;++dz)a+=SRC[e][dz][qy][qx]*MAT[qz*D1D+dz];dst[e][qz][qy][qx]=a;}
  ZST(g0,ix,B) ZST(g1,iy,B) ZST(g2,iz,G)
#undef ZST
  for(int e=0;e<NE;++e)for(int qz=0;qz<Q1D;++qz)for(int qy=0;qy<Q1D;++qy)for(int qx=0;qx<Q1D;++qx){int q=qx+Q1D*(qy+Q1D*qz);double a=g0[e][qz][qy][qx],b=g1[e][qz][qy][qx],c=g2[e][qz][qy][qx];h0[e][qz][qy][qx]=D[O3(q,0,e)]*a+D[O3(q,1,e)]*b+D[O3(q,2,e)]*c;h1[e][qz][qy][qx]=D[O3(q,1,e)]*a+D[O3(q,3,e)]*b+D[O3(q,4,e)]*c;h2[e][qz][qy][qx]=D[O3(q,2,e)]*a+D[O3(q,4,e)]*b+D[O3(q,5,e)]*c;}
#define XR(dst,SRC,MAT) for(int e=0;e<NE;++e)for(int qz=0;qz<Q1D;++qz)for(int qy=0;qy<Q1D;++qy)for(int dx=0;dx<D1D;++dx){double a=0.;for(int qx=0;qx<Q1D;++qx)a+=SRC[e][qz][qy][qx]*MAT[dx*Q1D+qx];dst[e][qz][qy][dx]=a;}
  XR(rx0,h0,Gt) XR(rx1,h1,Bt) XR(rx2,h2,Bt)
#undef XR
#define YR(dst,SRC,MAT) for(int e=0;e<NE;++e)for(int qz=0;qz<Q1D;++qz)for(int dy=0;dy<D1D;++dy)for(int dx=0;dx<D1D;++dx){double a=0.;for(int qy=0;qy<Q1D;++qy)a+=SRC[e][qz][qy][dx]*MAT[dy*Q1D+qy];dst[e][qz][dy][dx]=a;}
  YR(ry0,rx0,Bt) YR(ry1,rx1,Gt) YR(ry2,rx2,Bt)
#undef YR
#define ZR(dst,SRC,MAT) for(int e=0;e<NE;++e)for(int dz=0;dz<D1D;++dz)for(int dy=0;dy<D1D;++dy)for(int dx=0;dx<D1D;++dx){double a=0.;for(int qz=0;qz<Q1D;++qz)a+=SRC[e][qz][dy][dx]*MAT[dz*Q1D+qz];dst[e][dz][dy][dx]=a;}
  ZR(u0,ry0,Bt) ZR(u1,ry1,Bt) ZR(u2,ry2,Gt)
#undef ZR
  for(int e=0;e<NE;++e)for(int dz=0;dz<D1D;++dz)for(int dy=0;dy<D1D;++dy)for(int dx=0;dx<D1D;++dx)Y[X3(dx,dy,dz,e)]+=u0[e][dz][dy][dx]+u1[e][dz][dy][dx]+u2[e][dz][dy][dx];
}
