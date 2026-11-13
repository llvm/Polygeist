#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Use an old-style declaration at the call boundary because l2norm's VLA
// element type depends on ldx/ldy.  The ABI is still a flat pointer.
extern void l2norm();

int main(void) {
  const int ldx=6, ldy=5, ldz=4, nx=6, ny=5, nz=4;
  const int px=ldx/2*2+1, py=ldy/2*2+1;
  const int ist=1, iend=nx-1, jst=1, jend=ny-1;
  size_t count=(size_t)ldz*py*px*5;
  double *v=calloc(count,sizeof(double));
  double sum[5], expected[5]={0,0,0,0,0};
  if (!v) return 100;
  for (size_t i=0;i<count;++i) v[i]=0.001*(double)(i+1);
  for (int k=1;k<nz-1;++k)
    for (int j=jst;j<jend;++j)
      for (int i=ist;i<iend;++i)
        for (int m=0;m<5;++m) {
          double x=v[(((size_t)k*py+j)*px+i)*5+m];
          expected[m]+=x*x;
        }
  double denom=(double)((nx-2)*(ny-2)*(nz-2));
  for (int m=0;m<5;++m) expected[m]=sqrt(expected[m]/denom);
  l2norm(ldx,ldy,ldz,nx,ny,nz,ist,iend,jst,jend,v,sum);
  for (int m=0;m<5;++m) if (fabs(sum[m]-expected[m])>1e-11) {
    fprintf(stderr,"LU l2norm mismatch %d: %.17g vs %.17g\n",m,sum[m],expected[m]);
    return 1;
  }
  puts("npb-lu-l2norm: PASS");
  free(v);
  return 0;
}
