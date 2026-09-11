#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void psinv(signed char *, signed char *, int, int, int, double *, int);

static int at(int x, int y, int z, int n1, int n2) {
  return x + n1 * (y + n2 * z);
}

int main(void) {
  const int n1 = 6, n2 = 5, n3 = 4, n = n1*n2*n3;
  double *r = calloc((size_t)n, sizeof(double));
  double *u = calloc((size_t)n, sizeof(double));
  double *expected = calloc((size_t)n, sizeof(double));
  double c[4] = {0.3, -0.1, 0.05, 0.0};
  if (!r || !u || !expected) return 100;
  for (int i=0;i<n;++i) { r[i]=0.01*(i-2); u[i]=expected[i]=0.005*(i+1); }
  for (int z=1;z<n3-1;++z)
    for (int y=1;y<n2-1;++y)
      for (int x=1;x<n1-1;++x) {
        int p=at(x,y,z,n1,n2), sy=n1, sz=n1*n2;
        double r1=r[p-sy]+r[p+sy]+r[p-sz]+r[p+sz];
        double r1m=r[p-sy-1]+r[p+sy-1]+r[p-sz-1]+r[p+sz-1];
        double r1p=r[p-sy+1]+r[p+sy+1]+r[p-sz+1]+r[p+sz+1];
        double r2=r[p-sz-sy]+r[p-sz+sy]+r[p+sz-sy]+r[p+sz+sy];
        expected[p]+=c[0]*r[p]+c[1]*(r[p-1]+r[p+1]+r1)+c[2]*(r2+r1m+r1p);
      }
  psinv((signed char *)r,(signed char *)u,n1,n2,n3,c,0);
  for (int i=0;i<n;++i) if (fabs(u[i]-expected[i]) > 1e-11) {
    fprintf(stderr,"MG psinv mismatch %d: %.17g vs %.17g\n",i,u[i],expected[i]);
    return 1;
  }
  puts("npb-mg-psinv-core: PASS");
  free(r); free(u); free(expected);
  return 0;
}
