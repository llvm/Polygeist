/* Component-specialized, stage-sliced 2D H(curl) and H(div) operators. */
enum { D1D=4,Q1D=5,EDGE=3,NE=2,N2=24 };
#define V(v,e) ((v)+N2*(e))
#define O(qx,qy,e) ((qx)+Q1D*((qy)+Q1D*(e)))

void mfem_pa_curlcurl_apply_2d_stage_sliced(const double *Bo,const double *Bot,
 const double *Gc,const double *Gct,const double *op,const double *X,double *Y){
  double x0[NE][D1D][Q1D],x1[NE][EDGE][Q1D],c0[NE][Q1D][Q1D],c1[NE][Q1D][Q1D];
  double h[NE][Q1D][Q1D],r0[NE][Q1D][EDGE],r1[NE][Q1D][D1D];
  double u0[NE][D1D][EDGE],u1[NE][EDGE][D1D];
  for(int e=0;e<NE;++e)for(int dy=0;dy<D1D;++dy)for(int qx=0;qx<Q1D;++qx){double a=0.;for(int dx=0;dx<EDGE;++dx)a+=X[V(dx+EDGE*dy,e)]*Bo[qx*EDGE+dx];x0[e][dy][qx]=a;}
  for(int e=0;e<NE;++e)for(int qy=0;qy<Q1D;++qy)for(int qx=0;qx<Q1D;++qx){double a=0.;for(int dy=0;dy<D1D;++dy)a+=x0[e][dy][qx]*Gc[qy*D1D+dy];c0[e][qy][qx]=a;}
  for(int e=0;e<NE;++e)for(int dy=0;dy<EDGE;++dy)for(int qx=0;qx<Q1D;++qx){double a=0.;for(int dx=0;dx<D1D;++dx)a+=X[V(EDGE*D1D+dx+D1D*dy,e)]*Gc[qx*D1D+dx];x1[e][dy][qx]=a;}
  for(int e=0;e<NE;++e)for(int qy=0;qy<Q1D;++qy)for(int qx=0;qx<Q1D;++qx){double a=0.;for(int dy=0;dy<EDGE;++dy)a+=x1[e][dy][qx]*Bo[qy*EDGE+dy];c1[e][qy][qx]=a;}
  for(int e=0;e<NE;++e)for(int qy=0;qy<Q1D;++qy)for(int qx=0;qx<Q1D;++qx)h[e][qy][qx]=op[O(qx,qy,e)]*(c1[e][qy][qx]-c0[e][qy][qx]);
  for(int e=0;e<NE;++e)for(int qy=0;qy<Q1D;++qy)for(int dx=0;dx<EDGE;++dx){double a=0.;for(int qx=0;qx<Q1D;++qx)a+=h[e][qy][qx]*Bot[dx*Q1D+qx];r0[e][qy][dx]=a;}
  for(int e=0;e<NE;++e)for(int dy=0;dy<D1D;++dy)for(int dx=0;dx<EDGE;++dx){double a=0.;for(int qy=0;qy<Q1D;++qy)a-=r0[e][qy][dx]*Gct[dy*Q1D+qy];u0[e][dy][dx]=a;}
  for(int e=0;e<NE;++e)for(int qy=0;qy<Q1D;++qy)for(int dx=0;dx<D1D;++dx){double a=0.;for(int qx=0;qx<Q1D;++qx)a+=h[e][qy][qx]*Gct[dx*Q1D+qx];r1[e][qy][dx]=a;}
  for(int e=0;e<NE;++e)for(int dy=0;dy<EDGE;++dy)for(int dx=0;dx<D1D;++dx){double a=0.;for(int qy=0;qy<Q1D;++qy)a+=r1[e][qy][dx]*Bot[dy*Q1D+qy];u1[e][dy][dx]=a;}
  for(int e=0;e<NE;++e)for(int dy=0;dy<D1D;++dy)for(int dx=0;dx<EDGE;++dx)Y[V(dx+EDGE*dy,e)]+=u0[e][dy][dx];
  for(int e=0;e<NE;++e)for(int dy=0;dy<EDGE;++dy)for(int dx=0;dx<D1D;++dx)Y[V(EDGE*D1D+dx+D1D*dy,e)]+=u1[e][dy][dx];
}

void mfem_pa_divdiv_apply_2d_stage_sliced(const double *Bo,const double *Bot,
 const double *Gc,const double *Gct,const double *op,const double *X,double *Y){
  double x0[NE][EDGE][Q1D],x1[NE][D1D][Q1D],d0[NE][Q1D][Q1D],d1[NE][Q1D][Q1D];
  double h[NE][Q1D][Q1D],r0[NE][Q1D][D1D],r1[NE][Q1D][EDGE];
  double u0[NE][EDGE][D1D],u1[NE][D1D][EDGE];
  for(int e=0;e<NE;++e)for(int dy=0;dy<EDGE;++dy)for(int qx=0;qx<Q1D;++qx){double a=0.;for(int dx=0;dx<D1D;++dx)a+=X[V(dx+D1D*dy,e)]*Gc[qx*D1D+dx];x0[e][dy][qx]=a;}
  for(int e=0;e<NE;++e)for(int qy=0;qy<Q1D;++qy)for(int qx=0;qx<Q1D;++qx){double a=0.;for(int dy=0;dy<EDGE;++dy)a+=x0[e][dy][qx]*Bo[qy*EDGE+dy];d0[e][qy][qx]=a;}
  for(int e=0;e<NE;++e)for(int dy=0;dy<D1D;++dy)for(int qx=0;qx<Q1D;++qx){double a=0.;for(int dx=0;dx<EDGE;++dx)a+=X[V(D1D*EDGE+dx+EDGE*dy,e)]*Bo[qx*EDGE+dx];x1[e][dy][qx]=a;}
  for(int e=0;e<NE;++e)for(int qy=0;qy<Q1D;++qy)for(int qx=0;qx<Q1D;++qx){double a=0.;for(int dy=0;dy<D1D;++dy)a+=x1[e][dy][qx]*Gc[qy*D1D+dy];d1[e][qy][qx]=a;}
  for(int e=0;e<NE;++e)for(int qy=0;qy<Q1D;++qy)for(int qx=0;qx<Q1D;++qx)h[e][qy][qx]=op[O(qx,qy,e)]*(d0[e][qy][qx]+d1[e][qy][qx]);
  for(int e=0;e<NE;++e)for(int qy=0;qy<Q1D;++qy)for(int dx=0;dx<D1D;++dx){double a=0.;for(int qx=0;qx<Q1D;++qx)a+=h[e][qy][qx]*Gct[dx*Q1D+qx];r0[e][qy][dx]=a;}
  for(int e=0;e<NE;++e)for(int dy=0;dy<EDGE;++dy)for(int dx=0;dx<D1D;++dx){double a=0.;for(int qy=0;qy<Q1D;++qy)a+=r0[e][qy][dx]*Bot[dy*Q1D+qy];u0[e][dy][dx]=a;}
  for(int e=0;e<NE;++e)for(int qy=0;qy<Q1D;++qy)for(int dx=0;dx<EDGE;++dx){double a=0.;for(int qx=0;qx<Q1D;++qx)a+=h[e][qy][qx]*Bot[dx*Q1D+qx];r1[e][qy][dx]=a;}
  for(int e=0;e<NE;++e)for(int dy=0;dy<D1D;++dy)for(int dx=0;dx<EDGE;++dx){double a=0.;for(int qy=0;qy<Q1D;++qy)a+=r1[e][qy][dx]*Gct[dy*Q1D+qy];u1[e][dy][dx]=a;}
  for(int e=0;e<NE;++e)for(int dy=0;dy<EDGE;++dy)for(int dx=0;dx<D1D;++dx)Y[V(dx+D1D*dy,e)]+=u0[e][dy][dx];
  for(int e=0;e<NE;++e)for(int dy=0;dy<D1D;++dy)for(int dx=0;dx<EDGE;++dx)Y[V(D1D*EDGE+dx+EDGE*dy,e)]+=u1[e][dy][dx];
}
