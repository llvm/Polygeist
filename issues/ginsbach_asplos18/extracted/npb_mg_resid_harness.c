#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void resid(signed char *, signed char *, signed char *,
           int, int, int, double *, int);

static int at(int x, int y, int z, int n1, int n2) {
  return x + n1 * (y + n2 * z);
}

int main(void) {
  const int n1 = 6, n2 = 5, n3 = 4, n = n1*n2*n3;
  double *u = calloc((size_t)n, sizeof(double));
  double *v = calloc((size_t)n, sizeof(double));
  double *r = calloc((size_t)n, sizeof(double));
  double *expected = calloc((size_t)n, sizeof(double));
  double a[4] = {0.4, 0.0, -0.2, 0.1};
  if (!u || !v || !r || !expected) return 100;
  for (int i = 0; i < n; ++i) { u[i] = 0.01*(i+1); v[i] = 0.02*(i-3); }
  for (int z = 1; z < n3-1; ++z)
    for (int y = 1; y < n2-1; ++y)
      for (int x = 1; x < n1-1; ++x) {
        int p = at(x,y,z,n1,n2), sy=n1, sz=n1*n2;
        double u1m=u[p-sy-1]+u[p+sy-1]+u[p-sz-1]+u[p+sz-1];
        double u1p=u[p-sy+1]+u[p+sy+1]+u[p-sz+1]+u[p+sz+1];
        double u2=u[p-sz-sy]+u[p-sz+sy]+u[p+sz-sy]+u[p+sz+sy];
        double u2m=u[p-sz-sy-1]+u[p-sz+sy-1]+u[p+sz-sy-1]+u[p+sz+sy-1];
        double u2p=u[p-sz-sy+1]+u[p-sz+sy+1]+u[p+sz-sy+1]+u[p+sz+sy+1];
        expected[p]=v[p]-a[0]*u[p]-a[2]*(u2+u1m+u1p)-a[3]*(u2m+u2p);
      }
  resid((signed char *)u,(signed char *)v,(signed char *)r,n1,n2,n3,a,0);
  for (int i=0;i<n;++i) if (fabs(r[i]-expected[i]) > 1e-11) {
    fprintf(stderr,"MG resid mismatch %d: %.17g vs %.17g\n",i,r[i],expected[i]);
    return 1;
  }
  puts("npb-mg-resid-core: PASS");
  free(u); free(v); free(r); free(expected);
  return 0;
}
