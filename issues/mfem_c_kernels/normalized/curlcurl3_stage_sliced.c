/* Six-path component-specialized, stage-sliced 3D H(curl) operator. */
#ifndef MFEM_BENCH_NE
#define MFEM_BENCH_NE 2
#endif
enum { D1D=4,Q1D=5,EDGE=3,NE=MFEM_BENCH_NE,N3=144 };
#define V(v,e) ((v)+N3*(e))
#define OP(x,y,z,c,e) ((x)+Q1D*((y)+Q1D*((z)+Q1D*((c)+6*(e)))))
// polygeist-arg-extents mfem_pa_curlcurl_apply_3d_stage_sliced: Bo=15, Bc=20, Bot=15, Bct=20, Gc=20, Gct=20, op=750*MFEM_BENCH_NE, X=144*MFEM_BENCH_NE, Y=144*MFEM_BENCH_NE
void mfem_pa_curlcurl_apply_3d_stage_sliced(const double *Bo,const double *Bc,
 const double *Bot,const double *Bct,const double *Gc,const double *Gct,
 const double *op,const double *X,double *Y){
  double a0[NE][D1D][D1D][Q1D],a1x[NE][D1D][D1D][Q1D],a1b[NE][D1D][D1D][Q1D];
  double a2x[NE][D1D][D1D][Q1D],a2b[NE][D1D][D1D][Q1D];
  double b0y[NE][D1D][Q1D][Q1D],b0b[NE][D1D][Q1D][Q1D];
  double b1x[NE][D1D][Q1D][Q1D],b1z[NE][D1D][Q1D][Q1D];
  double b2x[NE][D1D][Q1D][Q1D],b2y[NE][D1D][Q1D][Q1D];
  double p0y[NE][Q1D][Q1D][Q1D],p0z[NE][Q1D][Q1D][Q1D];
  double p1x[NE][Q1D][Q1D][Q1D],p1z[NE][Q1D][Q1D][Q1D];
  double p2x[NE][Q1D][Q1D][Q1D],p2y[NE][Q1D][Q1D][Q1D];
  double h0[NE][Q1D][Q1D][Q1D],h1[NE][Q1D][Q1D][Q1D],h2[NE][Q1D][Q1D][Q1D];
#define XS(A,OFF,NX,NY,NZ,M) for(int e=0;e<NE;++e)for(int dz=0;dz<NZ;++dz)for(int dy=0;dy<NY;++dy)for(int qx=0;qx<Q1D;++qx){double v=0.;for(int dx=0;dx<NX;++dx)v+=X[V(OFF+dx+NX*(dy+NY*dz),e)]*M[qx*NX+dx];A[e][dz][dy][qx]=v;}
#define YS(B,A,NY,NZ,M) for(int e=0;e<NE;++e)for(int dz=0;dz<NZ;++dz)for(int qy=0;qy<Q1D;++qy)for(int qx=0;qx<Q1D;++qx){double v=0.;for(int dy=0;dy<NY;++dy)v+=A[e][dz][dy][qx]*M[qy*NY+dy];B[e][dz][qy][qx]=v;}
#define ZS(P,B,NZ,M) for(int e=0;e<NE;++e)for(int qz=0;qz<Q1D;++qz)for(int qy=0;qy<Q1D;++qy)for(int qx=0;qx<Q1D;++qx){double v=0.;for(int dz=0;dz<NZ;++dz)v+=B[e][dz][qy][qx]*M[qz*NZ+dz];P[e][qz][qy][qx]=v;}
  XS(a0,0,EDGE,D1D,D1D,Bo) YS(b0y,a0,D1D,D1D,Gc) YS(b0b,a0,D1D,D1D,Bc)
  ZS(p0y,b0y,D1D,Bc) ZS(p0z,b0b,D1D,Gc)
  const int o1=EDGE*D1D*D1D;
  XS(a1x,o1,D1D,EDGE,D1D,Gc) XS(a1b,o1,D1D,EDGE,D1D,Bc)
  YS(b1x,a1x,EDGE,D1D,Bo) YS(b1z,a1b,EDGE,D1D,Bo)
  ZS(p1x,b1x,D1D,Bc) ZS(p1z,b1z,D1D,Gc)
  const int o2=2*EDGE*D1D*D1D;
  XS(a2x,o2,D1D,D1D,EDGE,Gc) XS(a2b,o2,D1D,D1D,EDGE,Bc)
  YS(b2x,a2x,D1D,EDGE,Bc) YS(b2y,a2b,D1D,EDGE,Gc)
  ZS(p2x,b2x,EDGE,Bo) ZS(p2y,b2y,EDGE,Bo)
#undef XS
#undef YS
#undef ZS
  for(int e=0;e<NE;++e)for(int qz=0;qz<Q1D;++qz)for(int qy=0;qy<Q1D;++qy)for(int qx=0;qx<Q1D;++qx){double c0=p2y[e][qz][qy][qx]-p1z[e][qz][qy][qx],c1=p0z[e][qz][qy][qx]-p2x[e][qz][qy][qx],c2=p1x[e][qz][qy][qx]-p0y[e][qz][qy][qx];h0[e][qz][qy][qx]=op[OP(qx,qy,qz,0,e)]*c0+op[OP(qx,qy,qz,1,e)]*c1+op[OP(qx,qy,qz,2,e)]*c2;h1[e][qz][qy][qx]=op[OP(qx,qy,qz,1,e)]*c0+op[OP(qx,qy,qz,3,e)]*c1+op[OP(qx,qy,qz,4,e)]*c2;h2[e][qz][qy][qx]=op[OP(qx,qy,qz,2,e)]*c0+op[OP(qx,qy,qz,4,e)]*c1+op[OP(qx,qy,qz,5,e)]*c2;}
  double rx0z[NE][Q1D][Q1D][D1D],rx0y[NE][Q1D][Q1D][D1D];
  double rx1x[NE][Q1D][Q1D][D1D],rx1z[NE][Q1D][Q1D][D1D];
  double rx2y[NE][Q1D][Q1D][D1D],rx2x[NE][Q1D][Q1D][D1D];
  double ry0z[NE][Q1D][D1D][D1D],ry0y[NE][Q1D][D1D][D1D];
  double ry1x[NE][Q1D][D1D][D1D],ry1z[NE][Q1D][D1D][D1D];
  double ry2y[NE][Q1D][D1D][D1D],ry2x[NE][Q1D][D1D][D1D];
  double u0z[NE][D1D][D1D][D1D],u0y[NE][D1D][D1D][D1D];
  double u1x[NE][D1D][D1D][D1D],u1z[NE][D1D][D1D][D1D];
  double u2y[NE][D1D][D1D][D1D],u2x[NE][D1D][D1D][D1D];
#define XR(R,H,NX,M) for(int e=0;e<NE;++e)for(int qz=0;qz<Q1D;++qz)for(int qy=0;qy<Q1D;++qy)for(int dx=0;dx<NX;++dx){double v=0.;for(int qx=0;qx<Q1D;++qx)v+=H[e][qz][qy][qx]*M[dx*Q1D+qx];R[e][qz][qy][dx]=v;}
#define YR(S,R,NX,NY,M) for(int e=0;e<NE;++e)for(int qz=0;qz<Q1D;++qz)for(int dy=0;dy<NY;++dy)for(int dx=0;dx<NX;++dx){double v=0.;for(int qy=0;qy<Q1D;++qy)v+=R[e][qz][qy][dx]*M[dy*Q1D+qy];S[e][qz][dy][dx]=v;}
#define ZR(U,S,NX,NY,NZ,M) for(int e=0;e<NE;++e)for(int dz=0;dz<NZ;++dz)for(int dy=0;dy<NY;++dy)for(int dx=0;dx<NX;++dx){double v=0.;for(int qz=0;qz<Q1D;++qz)v+=S[e][qz][dy][dx]*M[dz*Q1D+qz];U[e][dz][dy][dx]=v;}
  XR(rx0z,h1,EDGE,Bot) YR(ry0z,rx0z,EDGE,D1D,Bct) ZR(u0z,ry0z,EDGE,D1D,D1D,Gct)
  XR(rx0y,h2,EDGE,Bot) YR(ry0y,rx0y,EDGE,D1D,Gct) ZR(u0y,ry0y,EDGE,D1D,D1D,Bct)
  XR(rx1x,h2,D1D,Gct) YR(ry1x,rx1x,D1D,EDGE,Bot) ZR(u1x,ry1x,D1D,EDGE,D1D,Bct)
  XR(rx1z,h0,D1D,Bct) YR(ry1z,rx1z,D1D,EDGE,Bot) ZR(u1z,ry1z,D1D,EDGE,D1D,Gct)
  XR(rx2y,h0,D1D,Bct) YR(ry2y,rx2y,D1D,D1D,Gct) ZR(u2y,ry2y,D1D,D1D,EDGE,Bot)
  XR(rx2x,h1,D1D,Gct) YR(ry2x,rx2x,D1D,D1D,Bct) ZR(u2x,ry2x,D1D,D1D,EDGE,Bot)
#undef XR
#undef YR
#undef ZR
  for(int e=0;e<NE;++e)for(int dz=0;dz<D1D;++dz)for(int dy=0;dy<D1D;++dy)for(int dx=0;dx<EDGE;++dx)Y[V(dx+EDGE*(dy+D1D*dz),e)]+=u0z[e][dz][dy][dx]-u0y[e][dz][dy][dx];
  for(int e=0;e<NE;++e)for(int dz=0;dz<D1D;++dz)for(int dy=0;dy<EDGE;++dy)for(int dx=0;dx<D1D;++dx)Y[V(o1+dx+D1D*(dy+EDGE*dz),e)]+=u1x[e][dz][dy][dx]-u1z[e][dz][dy][dx];
  for(int e=0;e<NE;++e)for(int dz=0;dz<EDGE;++dz)for(int dy=0;dy<D1D;++dy)for(int dx=0;dx<D1D;++dx)Y[V(o2+dx+D1D*(dy+D1D*dz),e)]+=u2y[e][dz][dy][dx]-u2x[e][dz][dy][dx];
}
