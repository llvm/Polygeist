/* Component-specialized, stage-sliced 3D H(div) div-div operator. */
#ifndef MFEM_BENCH_NE
#define MFEM_BENCH_NE 2
#endif
enum { D1D=4,Q1D=5,EDGE=3,NE=MFEM_BENCH_NE,N3=108 };
#define V(v,e) ((v)+N3*(e))
#define O(x,y,z,e) ((x)+Q1D*((y)+Q1D*((z)+Q1D*(e))))
// polygeist-arg-extents mfem_pa_divdiv_apply_3d_stage_sliced: Bo=15, Bot=15, Gc=20, Gct=20, op=125*MFEM_BENCH_NE, X=108*MFEM_BENCH_NE, Y=108*MFEM_BENCH_NE
void mfem_pa_divdiv_apply_3d_stage_sliced(const double *Bo,const double *Bot,
 const double *Gc,const double *Gct,const double *op,const double *X,double *Y){
  double a0[NE][EDGE][EDGE][Q1D];
  double a1[NE][EDGE][D1D][Q1D];
  double a2[NE][D1D][EDGE][Q1D];
  double b0[NE][EDGE][Q1D][Q1D];
  double b1[NE][EDGE][Q1D][Q1D];
  double b2[NE][D1D][Q1D][Q1D];
  double d0[NE][Q1D][Q1D][Q1D],d1[NE][Q1D][Q1D][Q1D],d2[NE][Q1D][Q1D][Q1D];
  double h[NE][Q1D][Q1D][Q1D];
  double r0[NE][Q1D][Q1D][D1D];
  double r1[NE][Q1D][Q1D][EDGE];
  double r2[NE][Q1D][Q1D][EDGE];
  /* These component paths have different H(div) edge extents.  Preserve the
   * actual loop domains in the scratch types instead of padding every axis to
   * D1D: Linalg otherwise infers a size-4 loop where the output view is 3. */
  double s0[NE][Q1D][EDGE][D1D];
  double s1[NE][Q1D][D1D][EDGE];
  double s2[NE][Q1D][EDGE][EDGE];
  double u0[NE][EDGE][EDGE][D1D];
  double u1[NE][EDGE][D1D][EDGE];
  double u2[NE][D1D][EDGE][EDGE];
  for(int e=0;e<NE;++e)for(int dz=0;dz<EDGE;++dz)for(int dy=0;dy<EDGE;++dy)for(int qx=0;qx<Q1D;++qx){double v=0.;for(int dx=0;dx<D1D;++dx)v+=X[V(dx+D1D*(dy+EDGE*dz),e)]*Gc[qx*D1D+dx];a0[e][dz][dy][qx]=v;}
  for(int e=0;e<NE;++e)for(int dz=0;dz<EDGE;++dz)for(int qy=0;qy<Q1D;++qy)for(int qx=0;qx<Q1D;++qx){double v=0.;for(int dy=0;dy<EDGE;++dy)v+=a0[e][dz][dy][qx]*Bo[qy*EDGE+dy];b0[e][dz][qy][qx]=v;}
  for(int e=0;e<NE;++e)for(int qz=0;qz<Q1D;++qz)for(int qy=0;qy<Q1D;++qy)for(int qx=0;qx<Q1D;++qx){double v=0.;for(int dz=0;dz<EDGE;++dz)v+=b0[e][dz][qy][qx]*Bo[qz*EDGE+dz];d0[e][qz][qy][qx]=v;}
  const int o1=D1D*EDGE*EDGE;
  for(int e=0;e<NE;++e)for(int dz=0;dz<EDGE;++dz)for(int dy=0;dy<D1D;++dy)for(int qx=0;qx<Q1D;++qx){double v=0.;for(int dx=0;dx<EDGE;++dx)v+=X[V(o1+dx+EDGE*(dy+D1D*dz),e)]*Bo[qx*EDGE+dx];a1[e][dz][dy][qx]=v;}
  for(int e=0;e<NE;++e)for(int dz=0;dz<EDGE;++dz)for(int qy=0;qy<Q1D;++qy)for(int qx=0;qx<Q1D;++qx){double v=0.;for(int dy=0;dy<D1D;++dy)v+=a1[e][dz][dy][qx]*Gc[qy*D1D+dy];b1[e][dz][qy][qx]=v;}
  for(int e=0;e<NE;++e)for(int qz=0;qz<Q1D;++qz)for(int qy=0;qy<Q1D;++qy)for(int qx=0;qx<Q1D;++qx){double v=0.;for(int dz=0;dz<EDGE;++dz)v+=b1[e][dz][qy][qx]*Bo[qz*EDGE+dz];d1[e][qz][qy][qx]=v;}
  const int o2=2*D1D*EDGE*EDGE;
  for(int e=0;e<NE;++e)for(int dz=0;dz<D1D;++dz)for(int dy=0;dy<EDGE;++dy)for(int qx=0;qx<Q1D;++qx){double v=0.;for(int dx=0;dx<EDGE;++dx)v+=X[V(o2+dx+EDGE*(dy+EDGE*dz),e)]*Bo[qx*EDGE+dx];a2[e][dz][dy][qx]=v;}
  for(int e=0;e<NE;++e)for(int dz=0;dz<D1D;++dz)for(int qy=0;qy<Q1D;++qy)for(int qx=0;qx<Q1D;++qx){double v=0.;for(int dy=0;dy<EDGE;++dy)v+=a2[e][dz][dy][qx]*Bo[qy*EDGE+dy];b2[e][dz][qy][qx]=v;}
  for(int e=0;e<NE;++e)for(int qz=0;qz<Q1D;++qz)for(int qy=0;qy<Q1D;++qy)for(int qx=0;qx<Q1D;++qx){double v=0.;for(int dz=0;dz<D1D;++dz)v+=b2[e][dz][qy][qx]*Gc[qz*D1D+dz];d2[e][qz][qy][qx]=v;}
  for(int e=0;e<NE;++e)for(int qz=0;qz<Q1D;++qz)for(int qy=0;qy<Q1D;++qy)for(int qx=0;qx<Q1D;++qx)h[e][qz][qy][qx]=op[O(qx,qy,qz,e)]*(d0[e][qz][qy][qx]+d1[e][qz][qy][qx]+d2[e][qz][qy][qx]);
#define XR(R,NX,M) for(int e=0;e<NE;++e)for(int qz=0;qz<Q1D;++qz)for(int qy=0;qy<Q1D;++qy)for(int dx=0;dx<NX;++dx){double v=0.;for(int qx=0;qx<Q1D;++qx)v+=h[e][qz][qy][qx]*M[dx*Q1D+qx];R[e][qz][qy][dx]=v;}
  XR(r0,D1D,Gct) XR(r1,EDGE,Bot) XR(r2,EDGE,Bot)
#undef XR
#define YR(S,R,NY,M) for(int e=0;e<NE;++e)for(int qz=0;qz<Q1D;++qz)for(int dy=0;dy<NY;++dy)for(int dx=0;dx<D1D;++dx){double v=0.;for(int qy=0;qy<Q1D;++qy)v+=R[e][qz][qy][dx]*M[dy*Q1D+qy];S[e][qz][dy][dx]=v;}
  YR(s0,r0,EDGE,Bot)
#undef YR
  for(int e=0;e<NE;++e)for(int qz=0;qz<Q1D;++qz)for(int dy=0;dy<D1D;++dy)for(int dx=0;dx<EDGE;++dx){double v=0.;for(int qy=0;qy<Q1D;++qy)v+=r1[e][qz][qy][dx]*Gct[dy*Q1D+qy];s1[e][qz][dy][dx]=v;}
  for(int e=0;e<NE;++e)for(int qz=0;qz<Q1D;++qz)for(int dy=0;dy<EDGE;++dy)for(int dx=0;dx<EDGE;++dx){double v=0.;for(int qy=0;qy<Q1D;++qy)v+=r2[e][qz][qy][dx]*Bot[dy*Q1D+qy];s2[e][qz][dy][dx]=v;}
  for(int e=0;e<NE;++e)for(int dz=0;dz<EDGE;++dz)for(int dy=0;dy<EDGE;++dy)for(int dx=0;dx<D1D;++dx){double v=0.;for(int qz=0;qz<Q1D;++qz)v+=s0[e][qz][dy][dx]*Bot[dz*Q1D+qz];u0[e][dz][dy][dx]=v;Y[V(dx+D1D*(dy+EDGE*dz),e)]+=v;}
  for(int e=0;e<NE;++e)for(int dz=0;dz<EDGE;++dz)for(int dy=0;dy<D1D;++dy)for(int dx=0;dx<EDGE;++dx){double v=0.;for(int qz=0;qz<Q1D;++qz)v+=s1[e][qz][dy][dx]*Bot[dz*Q1D+qz];u1[e][dz][dy][dx]=v;Y[V(o1+dx+EDGE*(dy+D1D*dz),e)]+=v;}
  for(int e=0;e<NE;++e)for(int dz=0;dz<D1D;++dz)for(int dy=0;dy<EDGE;++dy)for(int dx=0;dx<EDGE;++dx){double v=0.;for(int qz=0;qz<Q1D;++qz)v+=s2[e][qz][dy][dx]*Gct[dz*Q1D+qz];u2[e][dz][dy][dx]=v;Y[V(o2+dx+EDGE*(dy+EDGE*dz),e)]+=v;}
  (void)u0;(void)u1;(void)u2;
}
