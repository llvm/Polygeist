/*
 * Concrete C extraction of MFEM PACurlCurlApply3D, specialized to the
 * symmetric operator and standard signed curl (useAbs=false).
 */
enum { D1D=4, Q1D=5, EDGE=3, NE=2, NDOF=3*EDGE*D1D*D1D };
#define X(v,e) ((v)+NDOF*(e))
#define OP(qx,qy,qz,c,e) \
  ((qx)+Q1D*((qy)+Q1D*((qz)+Q1D*((c)+6*(e)))))

void mfem_pa_curlcurl_apply_3d(const double *Bo,const double *Bc,
                               const double *Bot,const double *Bct,
                               const double *Gc,const double *Gct,
                               const double *op,const double *x,double *y) {
  for(int e=0;e<NE;++e) {
    double curl[Q1D][Q1D][Q1D][3];
    for(int qz=0;qz<Q1D;++qz) for(int qy=0;qy<Q1D;++qy)
      for(int qx=0;qx<Q1D;++qx) for(int c=0;c<3;++c) curl[qz][qy][qx][c]=0.;

    for(int dz=0;dz<D1D;++dz) {
      double gxy[Q1D][Q1D][2];
      for(int qy=0;qy<Q1D;++qy) for(int qx=0;qx<Q1D;++qx)
        gxy[qy][qx][0]=gxy[qy][qx][1]=0.;
      for(int dy=0;dy<D1D;++dy) {
        double mx[Q1D]; for(int qx=0;qx<Q1D;++qx) mx[qx]=0.;
        for(int dx=0;dx<EDGE;++dx) for(int qx=0;qx<Q1D;++qx)
          mx[qx]+=x[X(dx+EDGE*(dy+D1D*dz),e)]*Bo[qx*EDGE+dx];
        for(int qy=0;qy<Q1D;++qy) for(int qx=0;qx<Q1D;++qx) {
          gxy[qy][qx][0]+=mx[qx]*Gc[qy*D1D+dy];
          gxy[qy][qx][1]+=mx[qx]*Bc[qy*D1D+dy];
        }
      }
      for(int qz=0;qz<Q1D;++qz) for(int qy=0;qy<Q1D;++qy)
        for(int qx=0;qx<Q1D;++qx) {
          curl[qz][qy][qx][1]+=gxy[qy][qx][1]*Gc[qz*D1D+dz];
          curl[qz][qy][qx][2]-=gxy[qy][qx][0]*Bc[qz*D1D+dz];
        }
    }
    const int off1=EDGE*D1D*D1D;
    for(int dz=0;dz<D1D;++dz) {
      double gxy[Q1D][Q1D][2];
      for(int qy=0;qy<Q1D;++qy) for(int qx=0;qx<Q1D;++qx)
        gxy[qy][qx][0]=gxy[qy][qx][1]=0.;
      for(int dx=0;dx<D1D;++dx) {
        double my[Q1D]; for(int qy=0;qy<Q1D;++qy) my[qy]=0.;
        for(int dy=0;dy<EDGE;++dy) for(int qy=0;qy<Q1D;++qy)
          my[qy]+=x[X(off1+dx+D1D*(dy+EDGE*dz),e)]*Bo[qy*EDGE+dy];
        for(int qx=0;qx<Q1D;++qx) for(int qy=0;qy<Q1D;++qy) {
          gxy[qy][qx][0]+=Gc[qx*D1D+dx]*my[qy];
          gxy[qy][qx][1]+=Bc[qx*D1D+dx]*my[qy];
        }
      }
      for(int qz=0;qz<Q1D;++qz) for(int qy=0;qy<Q1D;++qy)
        for(int qx=0;qx<Q1D;++qx) {
          curl[qz][qy][qx][0]-=gxy[qy][qx][1]*Gc[qz*D1D+dz];
          curl[qz][qy][qx][2]+=gxy[qy][qx][0]*Bc[qz*D1D+dz];
        }
    }
    const int off2=2*EDGE*D1D*D1D;
    for(int dx=0;dx<D1D;++dx) {
      double gyz[Q1D][Q1D][2];
      for(int qz=0;qz<Q1D;++qz) for(int qy=0;qy<Q1D;++qy)
        gyz[qz][qy][0]=gyz[qz][qy][1]=0.;
      for(int dy=0;dy<D1D;++dy) {
        double mz[Q1D]; for(int qz=0;qz<Q1D;++qz) mz[qz]=0.;
        for(int dz=0;dz<EDGE;++dz) for(int qz=0;qz<Q1D;++qz)
          mz[qz]+=x[X(off2+dx+D1D*(dy+D1D*dz),e)]*Bo[qz*EDGE+dz];
        for(int qy=0;qy<Q1D;++qy) for(int qz=0;qz<Q1D;++qz) {
          gyz[qz][qy][0]+=mz[qz]*Bc[qy*D1D+dy];
          gyz[qz][qy][1]+=mz[qz]*Gc[qy*D1D+dy];
        }
      }
      for(int qx=0;qx<Q1D;++qx) for(int qy=0;qy<Q1D;++qy)
        for(int qz=0;qz<Q1D;++qz) {
          curl[qz][qy][qx][0]+=gyz[qz][qy][1]*Bc[qx*D1D+dx];
          curl[qz][qy][qx][1]-=gyz[qz][qy][0]*Gc[qx*D1D+dx];
        }
    }

    for(int qz=0;qz<Q1D;++qz) for(int qy=0;qy<Q1D;++qy)
      for(int qx=0;qx<Q1D;++qx) {
        double a=curl[qz][qy][qx][0],b=curl[qz][qy][qx][1],c=curl[qz][qy][qx][2];
        double o11=op[OP(qx,qy,qz,0,e)],o12=op[OP(qx,qy,qz,1,e)];
        double o13=op[OP(qx,qy,qz,2,e)],o22=op[OP(qx,qy,qz,3,e)];
        double o23=op[OP(qx,qy,qz,4,e)],o33=op[OP(qx,qy,qz,5,e)];
        curl[qz][qy][qx][0]=o11*a+o12*b+o13*c;
        curl[qz][qy][qx][1]=o12*a+o22*b+o23*c;
        curl[qz][qy][qx][2]=o13*a+o23*b+o33*c;
      }

    for(int qz=0;qz<Q1D;++qz) {
      double g12[D1D][EDGE],g21[D1D][EDGE];
      for(int dy=0;dy<D1D;++dy) for(int dx=0;dx<EDGE;++dx) g12[dy][dx]=g21[dy][dx]=0.;
      for(int qy=0;qy<Q1D;++qy) {
        double mx[EDGE][2]; for(int dx=0;dx<EDGE;++dx) mx[dx][0]=mx[dx][1]=0.;
        for(int qx=0;qx<Q1D;++qx) for(int dx=0;dx<EDGE;++dx) {
          mx[dx][0]+=Bot[dx*Q1D+qx]*curl[qz][qy][qx][1];
          mx[dx][1]+=Bot[dx*Q1D+qx]*curl[qz][qy][qx][2];
        }
        for(int dy=0;dy<D1D;++dy) for(int dx=0;dx<EDGE;++dx) {
          g21[dy][dx]+=mx[dx][0]*Bct[dy*Q1D+qy];
          g12[dy][dx]+=mx[dx][1]*Gct[dy*Q1D+qy];
        }
      }
      for(int dz=0;dz<D1D;++dz) for(int dy=0;dy<D1D;++dy)
        for(int dx=0;dx<EDGE;++dx)
          y[X(dx+EDGE*(dy+D1D*dz),e)]+=g21[dy][dx]*Gct[dz*Q1D+qz]
                                                -g12[dy][dx]*Bct[dz*Q1D+qz];
    }
    for(int qz=0;qz<Q1D;++qz) {
      double g02[EDGE][D1D],g20[EDGE][D1D];
      for(int dy=0;dy<EDGE;++dy) for(int dx=0;dx<D1D;++dx) g02[dy][dx]=g20[dy][dx]=0.;
      for(int qx=0;qx<Q1D;++qx) {
        double my[EDGE][2]; for(int dy=0;dy<EDGE;++dy) my[dy][0]=my[dy][1]=0.;
        for(int qy=0;qy<Q1D;++qy) for(int dy=0;dy<EDGE;++dy) {
          my[dy][0]+=Bot[dy*Q1D+qy]*curl[qz][qy][qx][2];
          my[dy][1]+=Bot[dy*Q1D+qy]*curl[qz][qy][qx][0];
        }
        for(int dx=0;dx<D1D;++dx) for(int dy=0;dy<EDGE;++dy) {
          g02[dy][dx]+=my[dy][0]*Gct[dx*Q1D+qx];
          g20[dy][dx]+=my[dy][1]*Bct[dx*Q1D+qx];
        }
      }
      for(int dz=0;dz<D1D;++dz) for(int dy=0;dy<EDGE;++dy)
        for(int dx=0;dx<D1D;++dx)
          y[X(off1+dx+D1D*(dy+EDGE*dz),e)]+=-g20[dy][dx]*Gct[dz*Q1D+qz]
                                                     +g02[dy][dx]*Bct[dz*Q1D+qz];
    }
    for(int qx=0;qx<Q1D;++qx) {
      double g01[EDGE][D1D],g10[EDGE][D1D];
      for(int dz=0;dz<EDGE;++dz) for(int dy=0;dy<D1D;++dy) g01[dz][dy]=g10[dz][dy]=0.;
      for(int qy=0;qy<Q1D;++qy) {
        double mz[EDGE][2]; for(int dz=0;dz<EDGE;++dz) mz[dz][0]=mz[dz][1]=0.;
        for(int qz=0;qz<Q1D;++qz) for(int dz=0;dz<EDGE;++dz) {
          mz[dz][0]+=Bot[dz*Q1D+qz]*curl[qz][qy][qx][0];
          mz[dz][1]+=Bot[dz*Q1D+qz]*curl[qz][qy][qx][1];
        }
        for(int dy=0;dy<D1D;++dy) for(int dz=0;dz<EDGE;++dz) {
          g01[dz][dy]+=Bct[dy*Q1D+qy]*mz[dz][1];
          g10[dz][dy]+=Gct[dy*Q1D+qy]*mz[dz][0];
        }
      }
      for(int dx=0;dx<D1D;++dx) for(int dy=0;dy<D1D;++dy)
        for(int dz=0;dz<EDGE;++dz)
          y[X(off2+dx+D1D*(dy+D1D*dz),e)]+=g10[dz][dy]*Bct[dx*Q1D+qx]
                                                      -g01[dz][dy]*Gct[dx*Q1D+qx];
    }
  }
}
