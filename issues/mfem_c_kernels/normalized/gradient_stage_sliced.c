/* Gradient maps with disjoint component/stage scratch buffers. */
enum { D1D=4,Q1D=5,VDIM=2 };
#define F2(dx,dy,v) ((dx)+D1D*((dy)+D1D*(v)))
#define F3(dx,dy,dz,v) ((dx)+D1D*((dy)+D1D*((dz)+D1D*(v))))
#define Q2(v,c,qx,qy) ((qy)+Q1D*((qx)+Q1D*((c)+2*(v))))
#define Q3(v,c,qx,qy,qz) ((qz)+Q1D*((qy)+Q1D*((qx)+Q1D*((c)+3*(v)))))

void mfem_interp_grad_2d_stage_sliced(const double *f,const double *B,
                                       const double *G,double *o) {
  double sb[VDIM][D1D][Q1D],sg[VDIM][D1D][Q1D];
  for(int v=0;v<VDIM;++v) for(int dy=0;dy<D1D;++dy) for(int qx=0;qx<Q1D;++qx) {
    double a=0.; for(int dx=0;dx<D1D;++dx) a+=f[F2(dx,dy,v)]*B[qx*D1D+dx]; sb[v][dy][qx]=a; }
  for(int v=0;v<VDIM;++v) for(int dy=0;dy<D1D;++dy) for(int qx=0;qx<Q1D;++qx) {
    double a=0.; for(int dx=0;dx<D1D;++dx) a+=f[F2(dx,dy,v)]*G[qx*D1D+dx]; sg[v][dy][qx]=a; }
  for(int v=0;v<VDIM;++v) for(int qy=0;qy<Q1D;++qy) for(int qx=0;qx<Q1D;++qx) {
    double a=0.; for(int dy=0;dy<D1D;++dy) a+=sg[v][dy][qx]*B[qy*D1D+dy]; o[Q2(v,0,qx,qy)]=a; }
  for(int v=0;v<VDIM;++v) for(int qy=0;qy<Q1D;++qy) for(int qx=0;qx<Q1D;++qx) {
    double a=0.; for(int dy=0;dy<D1D;++dy) a+=sb[v][dy][qx]*G[qy*D1D+dy]; o[Q2(v,1,qx,qy)]=a; }
}

void mfem_interp_grad_3d_stage_sliced(const double *f,const double *B,
                                       const double *G,double *o) {
  double sb[VDIM][D1D][D1D][Q1D],sg[VDIM][D1D][D1D][Q1D];
  double sx[VDIM][D1D][Q1D][Q1D],sy[VDIM][D1D][Q1D][Q1D],sz[VDIM][D1D][Q1D][Q1D];
  for(int v=0;v<VDIM;++v) for(int dz=0;dz<D1D;++dz) for(int dy=0;dy<D1D;++dy)
    for(int qx=0;qx<Q1D;++qx) { double a=0.; for(int dx=0;dx<D1D;++dx) a+=f[F3(dx,dy,dz,v)]*B[qx*D1D+dx]; sb[v][dz][dy][qx]=a; }
  for(int v=0;v<VDIM;++v) for(int dz=0;dz<D1D;++dz) for(int dy=0;dy<D1D;++dy)
    for(int qx=0;qx<Q1D;++qx) { double a=0.; for(int dx=0;dx<D1D;++dx) a+=f[F3(dx,dy,dz,v)]*G[qx*D1D+dx]; sg[v][dz][dy][qx]=a; }
  for(int v=0;v<VDIM;++v) for(int dz=0;dz<D1D;++dz) for(int qy=0;qy<Q1D;++qy)
    for(int qx=0;qx<Q1D;++qx) { double a=0.; for(int dy=0;dy<D1D;++dy) a+=sg[v][dz][dy][qx]*B[qy*D1D+dy]; sx[v][dz][qy][qx]=a; }
  for(int v=0;v<VDIM;++v) for(int dz=0;dz<D1D;++dz) for(int qy=0;qy<Q1D;++qy)
    for(int qx=0;qx<Q1D;++qx) { double a=0.; for(int dy=0;dy<D1D;++dy) a+=sb[v][dz][dy][qx]*G[qy*D1D+dy]; sy[v][dz][qy][qx]=a; }
  for(int v=0;v<VDIM;++v) for(int dz=0;dz<D1D;++dz) for(int qy=0;qy<Q1D;++qy)
    for(int qx=0;qx<Q1D;++qx) { double a=0.; for(int dy=0;dy<D1D;++dy) a+=sb[v][dz][dy][qx]*B[qy*D1D+dy]; sz[v][dz][qy][qx]=a; }
  for(int v=0;v<VDIM;++v) for(int qz=0;qz<Q1D;++qz) for(int qy=0;qy<Q1D;++qy)
    for(int qx=0;qx<Q1D;++qx) { double a=0.; for(int dz=0;dz<D1D;++dz) a+=sx[v][dz][qy][qx]*B[qz*D1D+dz]; o[Q3(v,0,qx,qy,qz)]=a; }
  for(int v=0;v<VDIM;++v) for(int qz=0;qz<Q1D;++qz) for(int qy=0;qy<Q1D;++qy)
    for(int qx=0;qx<Q1D;++qx) { double a=0.; for(int dz=0;dz<D1D;++dz) a+=sy[v][dz][qy][qx]*B[qz*D1D+dz]; o[Q3(v,1,qx,qy,qz)]=a; }
  for(int v=0;v<VDIM;++v) for(int qz=0;qz<Q1D;++qz) for(int qy=0;qy<Q1D;++qy)
    for(int qx=0;qx<Q1D;++qx) { double a=0.; for(int dz=0;dz<D1D;++dz) a+=sz[v][dz][qy][qx]*G[qz*D1D+dz]; o[Q3(v,2,qx,qy,qz)]=a; }
}

void mfem_integrate_grad_2d_stage_sliced(const double *f,const double *B,
                                          const double *G,double *y) {
  double sx[VDIM][Q1D][D1D],sy[VDIM][Q1D][D1D],tx[VDIM][D1D][D1D],ty[VDIM][D1D][D1D];
  for(int v=0;v<VDIM;++v) for(int qy=0;qy<Q1D;++qy) for(int dx=0;dx<D1D;++dx) { double a=0.; for(int qx=0;qx<Q1D;++qx) a+=f[Q2(v,0,qx,qy)]*G[qx*D1D+dx]; sx[v][qy][dx]=a; }
  for(int v=0;v<VDIM;++v) for(int qy=0;qy<Q1D;++qy) for(int dx=0;dx<D1D;++dx) { double a=0.; for(int qx=0;qx<Q1D;++qx) a+=f[Q2(v,1,qx,qy)]*B[qx*D1D+dx]; sy[v][qy][dx]=a; }
  for(int v=0;v<VDIM;++v) for(int dy=0;dy<D1D;++dy) for(int dx=0;dx<D1D;++dx) { double a=0.; for(int qy=0;qy<Q1D;++qy) a+=sx[v][qy][dx]*B[qy*D1D+dy]; tx[v][dy][dx]=a; }
  for(int v=0;v<VDIM;++v) for(int dy=0;dy<D1D;++dy) for(int dx=0;dx<D1D;++dx) { double a=0.; for(int qy=0;qy<Q1D;++qy) a+=sy[v][qy][dx]*G[qy*D1D+dy]; ty[v][dy][dx]=a; }
  for(int v=0;v<VDIM;++v) for(int dy=0;dy<D1D;++dy) for(int dx=0;dx<D1D;++dx) y[F2(dx,dy,v)]+=tx[v][dy][dx]+ty[v][dy][dx];
}

void mfem_integrate_grad_3d_stage_sliced(const double *f,const double *B,
                                          const double *G,double *y) {
  double sx[VDIM][Q1D][Q1D][D1D],sy[VDIM][Q1D][Q1D][D1D],sz[VDIM][Q1D][Q1D][D1D];
  double tx[VDIM][Q1D][D1D][D1D],ty[VDIM][Q1D][D1D][D1D],tz[VDIM][Q1D][D1D][D1D];
  double ux[VDIM][D1D][D1D][D1D],uy[VDIM][D1D][D1D][D1D],uz[VDIM][D1D][D1D][D1D];
  for(int v=0;v<VDIM;++v) for(int qz=0;qz<Q1D;++qz) for(int qy=0;qy<Q1D;++qy) for(int dx=0;dx<D1D;++dx) { double a=0.; for(int qx=0;qx<Q1D;++qx) a+=f[Q3(v,0,qx,qy,qz)]*G[qx*D1D+dx]; sx[v][qz][qy][dx]=a; }
  for(int v=0;v<VDIM;++v) for(int qz=0;qz<Q1D;++qz) for(int qy=0;qy<Q1D;++qy) for(int dx=0;dx<D1D;++dx) { double a=0.; for(int qx=0;qx<Q1D;++qx) a+=f[Q3(v,1,qx,qy,qz)]*B[qx*D1D+dx]; sy[v][qz][qy][dx]=a; }
  for(int v=0;v<VDIM;++v) for(int qz=0;qz<Q1D;++qz) for(int qy=0;qy<Q1D;++qy) for(int dx=0;dx<D1D;++dx) { double a=0.; for(int qx=0;qx<Q1D;++qx) a+=f[Q3(v,2,qx,qy,qz)]*B[qx*D1D+dx]; sz[v][qz][qy][dx]=a; }
  for(int v=0;v<VDIM;++v) for(int qz=0;qz<Q1D;++qz) for(int dy=0;dy<D1D;++dy) for(int dx=0;dx<D1D;++dx) { double a=0.; for(int qy=0;qy<Q1D;++qy) a+=sx[v][qz][qy][dx]*B[qy*D1D+dy]; tx[v][qz][dy][dx]=a; }
  for(int v=0;v<VDIM;++v) for(int qz=0;qz<Q1D;++qz) for(int dy=0;dy<D1D;++dy) for(int dx=0;dx<D1D;++dx) { double a=0.; for(int qy=0;qy<Q1D;++qy) a+=sy[v][qz][qy][dx]*G[qy*D1D+dy]; ty[v][qz][dy][dx]=a; }
  for(int v=0;v<VDIM;++v) for(int qz=0;qz<Q1D;++qz) for(int dy=0;dy<D1D;++dy) for(int dx=0;dx<D1D;++dx) { double a=0.; for(int qy=0;qy<Q1D;++qy) a+=sz[v][qz][qy][dx]*B[qy*D1D+dy]; tz[v][qz][dy][dx]=a; }
  for(int v=0;v<VDIM;++v) for(int dz=0;dz<D1D;++dz) for(int dy=0;dy<D1D;++dy) for(int dx=0;dx<D1D;++dx) { double a=0.; for(int qz=0;qz<Q1D;++qz) a+=tx[v][qz][dy][dx]*B[qz*D1D+dz]; ux[v][dz][dy][dx]=a; }
  for(int v=0;v<VDIM;++v) for(int dz=0;dz<D1D;++dz) for(int dy=0;dy<D1D;++dy) for(int dx=0;dx<D1D;++dx) { double a=0.; for(int qz=0;qz<Q1D;++qz) a+=ty[v][qz][dy][dx]*B[qz*D1D+dz]; uy[v][dz][dy][dx]=a; }
  for(int v=0;v<VDIM;++v) for(int dz=0;dz<D1D;++dz) for(int dy=0;dy<D1D;++dy) for(int dx=0;dx<D1D;++dx) { double a=0.; for(int qz=0;qz<Q1D;++qz) a+=tz[v][qz][dy][dx]*G[qz*D1D+dz]; uz[v][dz][dy][dx]=a; }
  for(int v=0;v<VDIM;++v) for(int dz=0;dz<D1D;++dz) for(int dy=0;dy<D1D;++dy) for(int dx=0;dx<D1D;++dx) y[F3(dx,dy,dz,v)]+=ux[v][dz][dy][dx]+uy[v][dz][dy][dx]+uz[v][dz][dy][dx];
}
