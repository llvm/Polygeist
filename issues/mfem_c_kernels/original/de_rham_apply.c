/* Concrete C extraction of representative MFEM H(curl) and H(div) kernels. */
enum { D1D=4, Q1D=5, NE=2, EDGE=D1D-1, N2=2*EDGE*D1D,
       N3=3*EDGE*EDGE*D1D };
#define V2(v,e) ((v)+N2*(e))
#define V3(v,e) ((v)+N3*(e))
#define O2(qx,qy,e) ((qx)+Q1D*((qy)+Q1D*(e)))
#define O3(qx,qy,qz,e) ((qx)+Q1D*((qy)+Q1D*((qz)+Q1D*(e))))

void mfem_pa_curlcurl_apply_2d(const double *Bo, const double *Bot,
                               const double *Gc, const double *Gct,
                               const double *op, const double *x, double *y) {
  for (int e=0;e<NE;++e) {
    double curl[Q1D][Q1D];
    for(int qy=0;qy<Q1D;++qy) for(int qx=0;qx<Q1D;++qx) curl[qy][qx]=0.0;
    int osc=0;
    for(int c=0;c<2;++c) {
      int ny=(c==1)?EDGE:D1D, nx=(c==0)?EDGE:D1D;
      for(int dy=0;dy<ny;++dy) {
        double gx[Q1D]; for(int qx=0;qx<Q1D;++qx) gx[qx]=0.0;
        for(int dx=0;dx<nx;++dx) for(int qx=0;qx<Q1D;++qx)
          gx[qx]+=x[V2(dx+dy*nx+osc,e)]*((c==0)?Bo[qx*EDGE+dx]:Gc[qx*D1D+dx]);
        for(int qy=0;qy<Q1D;++qy) {
          double wy=(c==0)?-Gc[qy*D1D+dy]:Bo[qy*EDGE+dy];
          for(int qx=0;qx<Q1D;++qx) curl[qy][qx]+=gx[qx]*wy;
        }
      }
      osc+=nx*ny;
    }
    for(int qy=0;qy<Q1D;++qy) for(int qx=0;qx<Q1D;++qx)
      curl[qy][qx]*=op[O2(qx,qy,e)];
    for(int qy=0;qy<Q1D;++qy) {
      osc=0;
      for(int c=0;c<2;++c) {
        int ny=(c==1)?EDGE:D1D, nx=(c==0)?EDGE:D1D;
        double gx[D1D]; for(int dx=0;dx<nx;++dx) gx[dx]=0.0;
        for(int qx=0;qx<Q1D;++qx) for(int dx=0;dx<nx;++dx)
          gx[dx]+=curl[qy][qx]*((c==0)?Bot[dx*Q1D+qx]:Gct[dx*Q1D+qx]);
        for(int dy=0;dy<ny;++dy) {
          double wy=(c==0)?-Gct[dy*Q1D+qy]:Bot[dy*Q1D+qy];
          for(int dx=0;dx<nx;++dx) y[V2(dx+dy*nx+osc,e)]+=gx[dx]*wy;
        }
        osc+=nx*ny;
      }
    }
  }
}

void mfem_pa_divdiv_apply_2d(const double *Bo, const double *Bot,
                             const double *Gc, const double *Gct,
                             const double *op, const double *x, double *y) {
  for (int e=0;e<NE;++e) {
    double div[Q1D][Q1D];
    for(int qy=0;qy<Q1D;++qy) for(int qx=0;qx<Q1D;++qx) div[qy][qx]=0.0;
    int osc=0;
    for(int c=0;c<2;++c) {
      int nx=(c==1)?EDGE:D1D, ny=(c==0)?EDGE:D1D;
      for(int dy=0;dy<ny;++dy) {
        double gx[Q1D]; for(int qx=0;qx<Q1D;++qx) gx[qx]=0.0;
        for(int dx=0;dx<nx;++dx) for(int qx=0;qx<Q1D;++qx)
          gx[qx]+=x[V2(dx+dy*nx+osc,e)]*((c==0)?Gc[qx*D1D+dx]:Bo[qx*EDGE+dx]);
        for(int qy=0;qy<Q1D;++qy) {
          double wy=(c==0)?Bo[qy*EDGE+dy]:Gc[qy*D1D+dy];
          for(int qx=0;qx<Q1D;++qx) div[qy][qx]+=gx[qx]*wy;
        }
      }
      osc+=nx*ny;
    }
    for(int qy=0;qy<Q1D;++qy) for(int qx=0;qx<Q1D;++qx)
      div[qy][qx]*=op[O2(qx,qy,e)];
    for(int qy=0;qy<Q1D;++qy) {
      osc=0;
      for(int c=0;c<2;++c) {
        int nx=(c==1)?EDGE:D1D, ny=(c==0)?EDGE:D1D;
        double gx[D1D]; for(int dx=0;dx<nx;++dx) gx[dx]=0.0;
        for(int qx=0;qx<Q1D;++qx) for(int dx=0;dx<nx;++dx)
          gx[dx]+=div[qy][qx]*((c==0)?Gct[dx*Q1D+qx]:Bot[dx*Q1D+qx]);
        for(int dy=0;dy<ny;++dy) {
          double wy=(c==0)?Bot[dy*Q1D+qy]:Gct[dy*Q1D+qy];
          for(int dx=0;dx<nx;++dx) y[V2(dx+dy*nx+osc,e)]+=gx[dx]*wy;
        }
        osc+=nx*ny;
      }
    }
  }
}

void mfem_pa_divdiv_apply_3d(const double *Bo, const double *Bot,
                             const double *Gc, const double *Gct,
                             const double *op, const double *x, double *y) {
  for(int e=0;e<NE;++e) {
    double div[Q1D][Q1D][Q1D];
    for(int qz=0;qz<Q1D;++qz) for(int qy=0;qy<Q1D;++qy)
      for(int qx=0;qx<Q1D;++qx) div[qz][qy][qx]=0.0;
    int osc=0;
    for(int c=0;c<3;++c) {
      int nz=(c==2)?D1D:EDGE, ny=(c==1)?D1D:EDGE, nx=(c==0)?D1D:EDGE;
      for(int dz=0;dz<nz;++dz) {
        double axy[Q1D][Q1D];
        for(int qy=0;qy<Q1D;++qy) for(int qx=0;qx<Q1D;++qx) axy[qy][qx]=0.0;
        for(int dy=0;dy<ny;++dy) {
          double ax[Q1D]; for(int qx=0;qx<Q1D;++qx) ax[qx]=0.0;
          for(int dx=0;dx<nx;++dx) for(int qx=0;qx<Q1D;++qx)
            ax[qx]+=x[V3(dx+(dy+dz*ny)*nx+osc,e)]*
                    ((c==0)?Gc[qx*D1D+dx]:Bo[qx*EDGE+dx]);
          for(int qy=0;qy<Q1D;++qy) for(int qx=0;qx<Q1D;++qx)
            axy[qy][qx]+=ax[qx]*((c==1)?Gc[qy*D1D+dy]:Bo[qy*EDGE+dy]);
        }
        for(int qz=0;qz<Q1D;++qz) for(int qy=0;qy<Q1D;++qy)
          for(int qx=0;qx<Q1D;++qx)
            div[qz][qy][qx]+=axy[qy][qx]*((c==2)?Gc[qz*D1D+dz]:Bo[qz*EDGE+dz]);
      }
      osc+=nx*ny*nz;
    }
    for(int qz=0;qz<Q1D;++qz) for(int qy=0;qy<Q1D;++qy)
      for(int qx=0;qx<Q1D;++qx) div[qz][qy][qx]*=op[O3(qx,qy,qz,e)];
    for(int qz=0;qz<Q1D;++qz) {
      osc=0;
      for(int c=0;c<3;++c) {
        int nz=(c==2)?D1D:EDGE, ny=(c==1)?D1D:EDGE, nx=(c==0)?D1D:EDGE;
        double axy[D1D][D1D];
        for(int dy=0;dy<ny;++dy) for(int dx=0;dx<nx;++dx) axy[dy][dx]=0.0;
        for(int qy=0;qy<Q1D;++qy) {
          double ax[D1D]; for(int dx=0;dx<nx;++dx) ax[dx]=0.0;
          for(int qx=0;qx<Q1D;++qx) for(int dx=0;dx<nx;++dx)
            ax[dx]+=div[qz][qy][qx]*((c==0)?Gct[dx*Q1D+qx]:Bot[dx*Q1D+qx]);
          for(int dy=0;dy<ny;++dy) for(int dx=0;dx<nx;++dx)
            axy[dy][dx]+=ax[dx]*((c==1)?Gct[dy*Q1D+qy]:Bot[dy*Q1D+qy]);
        }
        for(int dz=0;dz<nz;++dz) for(int dy=0;dy<ny;++dy)
          for(int dx=0;dx<nx;++dx)
            y[V3(dx+(dy+dz*ny)*nx+osc,e)]+=axy[dy][dx]*
              ((c==2)?Gct[dz*Q1D+qz]:Bot[dz*Q1D+qz]);
        osc+=nx*ny*nz;
      }
    }
  }
}
