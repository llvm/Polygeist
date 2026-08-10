/* Mass apply with disjoint element/stage scratch instead of sequential reuse. */
#ifndef MFEM_BENCH_NE
#define MFEM_BENCH_NE 2
#endif
enum { D1D=4, Q1D=5, NE=MFEM_BENCH_NE };
#define X2(dx,dy,e) ((dx)+D1D*((dy)+D1D*(e)))
#define D2(qx,qy,e) ((qx)+Q1D*((qy)+Q1D*(e)))
#define X3(dx,dy,dz,e) ((dx)+D1D*((dy)+D1D*((dz)+D1D*(e))))
#define D3(qx,qy,qz,e) ((qx)+Q1D*((qy)+Q1D*((qz)+Q1D*(e))))

// polygeist-arg-extents mfem_pa_mass_apply_2d_stage_sliced: B=20, Bt=20, D=25*MFEM_BENCH_NE, X=16*MFEM_BENCH_NE, Y=16*MFEM_BENCH_NE
void mfem_pa_mass_apply_2d_stage_sliced(const double *B,const double *Bt,
                                         const double *D,const double *X,double *Y) {
  double sx[NE][D1D][Q1D], sq[NE][Q1D][Q1D], tx[NE][Q1D][D1D];
  for(int e=0;e<NE;++e) for(int dy=0;dy<D1D;++dy) for(int qx=0;qx<Q1D;++qx) {
    double a=0.; for(int dx=0;dx<D1D;++dx) a+=B[qx*D1D+dx]*X[X2(dx,dy,e)];
    sx[e][dy][qx]=a;
  }
  for(int e=0;e<NE;++e) for(int qy=0;qy<Q1D;++qy) for(int qx=0;qx<Q1D;++qx) {
    double a=0.; for(int dy=0;dy<D1D;++dy) a+=B[qy*D1D+dy]*sx[e][dy][qx];
    sq[e][qy][qx]=a;
  }
  for(int e=0;e<NE;++e) for(int qy=0;qy<Q1D;++qy) for(int qx=0;qx<Q1D;++qx)
    sq[e][qy][qx]*=D[D2(qx,qy,e)];
  for(int e=0;e<NE;++e) for(int qy=0;qy<Q1D;++qy) for(int dx=0;dx<D1D;++dx) {
    double a=0.; for(int qx=0;qx<Q1D;++qx) a+=Bt[dx*Q1D+qx]*sq[e][qy][qx];
    tx[e][qy][dx]=a;
  }
  for(int e=0;e<NE;++e) for(int dy=0;dy<D1D;++dy) for(int dx=0;dx<D1D;++dx) {
    double a=0.; for(int qy=0;qy<Q1D;++qy) a+=Bt[dy*Q1D+qy]*tx[e][qy][dx];
    Y[X2(dx,dy,e)]+=a;
  }
}

// polygeist-arg-extents mfem_pa_mass_apply_3d_stage_sliced: B=20, Bt=20, D=125*MFEM_BENCH_NE, X=64*MFEM_BENCH_NE, Y=64*MFEM_BENCH_NE
void mfem_pa_mass_apply_3d_stage_sliced(const double *B,const double *Bt,
                                         const double *D,const double *X,double *Y) {
  double sx[NE][D1D][D1D][Q1D], sxy[NE][D1D][Q1D][Q1D];
  double sq[NE][Q1D][Q1D][Q1D], tx[NE][Q1D][Q1D][D1D];
  double txy[NE][Q1D][D1D][D1D];
  for(int e=0;e<NE;++e) for(int dz=0;dz<D1D;++dz) for(int dy=0;dy<D1D;++dy)
    for(int qx=0;qx<Q1D;++qx) { double a=0.; for(int dx=0;dx<D1D;++dx) {
      a+=B[qx*D1D+dx]*X[X3(dx,dy,dz,e)]; } sx[e][dz][dy][qx]=a; }
  for(int e=0;e<NE;++e) for(int dz=0;dz<D1D;++dz) for(int qy=0;qy<Q1D;++qy)
    for(int qx=0;qx<Q1D;++qx) { double a=0.; for(int dy=0;dy<D1D;++dy) {
      a+=B[qy*D1D+dy]*sx[e][dz][dy][qx]; } sxy[e][dz][qy][qx]=a; }
  for(int e=0;e<NE;++e) for(int qz=0;qz<Q1D;++qz) for(int qy=0;qy<Q1D;++qy)
    for(int qx=0;qx<Q1D;++qx) { double a=0.; for(int dz=0;dz<D1D;++dz) {
      a+=B[qz*D1D+dz]*sxy[e][dz][qy][qx]; } sq[e][qz][qy][qx]=a; }
  for(int e=0;e<NE;++e) for(int qz=0;qz<Q1D;++qz) for(int qy=0;qy<Q1D;++qy)
    for(int qx=0;qx<Q1D;++qx) sq[e][qz][qy][qx]*=D[D3(qx,qy,qz,e)];
  for(int e=0;e<NE;++e) for(int qz=0;qz<Q1D;++qz) for(int qy=0;qy<Q1D;++qy)
    for(int dx=0;dx<D1D;++dx) { double a=0.; for(int qx=0;qx<Q1D;++qx) {
      a+=Bt[dx*Q1D+qx]*sq[e][qz][qy][qx]; } tx[e][qz][qy][dx]=a; }
  for(int e=0;e<NE;++e) for(int qz=0;qz<Q1D;++qz) for(int dy=0;dy<D1D;++dy)
    for(int dx=0;dx<D1D;++dx) { double a=0.; for(int qy=0;qy<Q1D;++qy) {
      a+=Bt[dy*Q1D+qy]*tx[e][qz][qy][dx]; } txy[e][qz][dy][dx]=a; }
  for(int e=0;e<NE;++e) for(int dz=0;dz<D1D;++dz) for(int dy=0;dy<D1D;++dy)
    for(int dx=0;dx<D1D;++dx) { double a=0.; for(int qz=0;qz<Q1D;++qz) {
      a+=Bt[dz*Q1D+qz]*txy[e][qz][dy][dx]; } Y[X3(dx,dy,dz,e)]+=a; }
}
