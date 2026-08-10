/*
 * Normalized value interpolation/integration variants.
 * Unlike MFEM's GPU-oriented sequential reuse of scratch for each component,
 * these give every component a disjoint scratch slice.  This removes the
 * artificial cross-component memory dependence without changing arithmetic.
 */
#ifndef MFEM_BENCH_NE
#define MFEM_BENCH_NE 2
#endif
enum { D1D=4, Q1D=5, VDIM=MFEM_BENCH_NE };
#define F2(dx,dy,v) ((dx)+D1D*((dy)+D1D*(v)))
#define F3(dx,dy,dz,v) ((dx)+D1D*((dy)+D1D*((dz)+D1D*(v))))
#define Q2(v,qx,qy) ((qy)+Q1D*((qx)+Q1D*(v)))
#define Q3(v,qx,qy,qz) ((qz)+Q1D*((qy)+Q1D*((qx)+Q1D*(v))))

// polygeist-arg-extents mfem_interp_value_2d_scratch_sliced: field=16*MFEM_BENCH_NE, B=20, out=25*MFEM_BENCH_NE
void mfem_interp_value_2d_scratch_sliced(const double *field,
                                          const double *B, double *out) {
  double s0[VDIM][D1D][Q1D];
  for(int v=0;v<VDIM;++v) for(int dy=0;dy<D1D;++dy) for(int qx=0;qx<Q1D;++qx) {
    double a=0.0; for(int dx=0;dx<D1D;++dx) a+=B[qx*D1D+dx]*field[F2(dx,dy,v)];
    s0[v][dy][qx]=a;
  }
  for(int v=0;v<VDIM;++v) for(int qx=0;qx<Q1D;++qx) for(int qy=0;qy<Q1D;++qy) {
    double a=0.0; for(int dy=0;dy<D1D;++dy) a+=s0[v][dy][qx]*B[qy*D1D+dy];
    out[Q2(v,qx,qy)]=a;
  }
}

// polygeist-arg-extents mfem_interp_value_3d_scratch_sliced: field=64*MFEM_BENCH_NE, B=20, out=125*MFEM_BENCH_NE
void mfem_interp_value_3d_scratch_sliced(const double *field,
                                          const double *B, double *out) {
  double s0[VDIM][D1D][D1D][Q1D], s1[VDIM][D1D][Q1D][Q1D];
  for(int v=0;v<VDIM;++v) for(int dz=0;dz<D1D;++dz) for(int dy=0;dy<D1D;++dy)
    for(int qx=0;qx<Q1D;++qx) { double a=0.0; for(int dx=0;dx<D1D;++dx) {
      a+=B[qx*D1D+dx]*field[F3(dx,dy,dz,v)]; } s0[v][dz][dy][qx]=a; }
  for(int v=0;v<VDIM;++v) for(int dz=0;dz<D1D;++dz) for(int qx=0;qx<Q1D;++qx)
    for(int qy=0;qy<Q1D;++qy) { double a=0.0; for(int dy=0;dy<D1D;++dy) {
      a+=s0[v][dz][dy][qx]*B[qy*D1D+dy]; } s1[v][dz][qy][qx]=a; }
  for(int v=0;v<VDIM;++v) for(int qz=0;qz<Q1D;++qz) for(int qy=0;qy<Q1D;++qy)
    for(int qx=0;qx<Q1D;++qx) { double a=0.0; for(int dz=0;dz<D1D;++dz) {
      a+=s1[v][dz][qy][qx]*B[qz*D1D+dz]; } out[Q3(v,qx,qy,qz)]=a; }
}

// polygeist-arg-extents mfem_integrate_value_2d_scratch_sliced: fqp=25*MFEM_BENCH_NE, B=20, y=16*MFEM_BENCH_NE
void mfem_integrate_value_2d_scratch_sliced(const double *fqp,
                                             const double *B, double *y) {
  double s0[VDIM][Q1D][D1D];
  for(int v=0;v<VDIM;++v) for(int qy=0;qy<Q1D;++qy) for(int dx=0;dx<D1D;++dx) {
    double a=0.0; for(int qx=0;qx<Q1D;++qx) a+=fqp[Q2(v,qx,qy)]*B[qx*D1D+dx];
    s0[v][qy][dx]=a;
  }
  for(int v=0;v<VDIM;++v) for(int dy=0;dy<D1D;++dy) for(int dx=0;dx<D1D;++dx) {
    double a=0.0; for(int qy=0;qy<Q1D;++qy) a+=s0[v][qy][dx]*B[qy*D1D+dy];
    y[F2(dx,dy,v)]+=a;
  }
}

// polygeist-arg-extents mfem_integrate_value_3d_scratch_sliced: fqp=125*MFEM_BENCH_NE, B=20, y=64*MFEM_BENCH_NE
void mfem_integrate_value_3d_scratch_sliced(const double *fqp,
                                             const double *B, double *y) {
  double s0[VDIM][Q1D][Q1D][D1D], s1[VDIM][Q1D][D1D][D1D];
  for(int v=0;v<VDIM;++v) for(int qy=0;qy<Q1D;++qy) for(int dx=0;dx<D1D;++dx)
    for(int qz=0;qz<Q1D;++qz) { double a=0.0; for(int qx=0;qx<Q1D;++qx) {
      a+=fqp[Q3(v,qx,qy,qz)]*B[qx*D1D+dx]; } s0[v][qz][qy][dx]=a; }
  for(int v=0;v<VDIM;++v) for(int dy=0;dy<D1D;++dy) for(int dx=0;dx<D1D;++dx)
    for(int qz=0;qz<Q1D;++qz) { double a=0.0; for(int qy=0;qy<Q1D;++qy) {
      a+=s0[v][qz][qy][dx]*B[qy*D1D+dy]; } s1[v][qz][dy][dx]=a; }
  for(int v=0;v<VDIM;++v) for(int dz=0;dz<D1D;++dz) for(int dy=0;dy<D1D;++dy)
    for(int dx=0;dx<D1D;++dx) { double a=0.0; for(int qz=0;qz<Q1D;++qz) {
      a+=s1[v][qz][dy][dx]*B[qz*D1D+dz]; } y[F3(dx,dy,dz,v)]+=a; }
}
