#include <stdio.h>
#include <stdlib.h>

struct cartesian { float x, y, z; };
int doCompute(struct cartesian *, int, struct cartesian *, int, int,
              long long *, int, float *);

static void reference(struct cartesian *a, int n1, struct cartesian *b, int n2,
                      int self, long long *bins, int nbins, float *bounds) {
  if (self) { b=a; n2=n1; }
  for (int i=0;i<(self?n1-1:n1);++i)
    for (int j=self?i+1:0;j<n2;++j) {
      float dot=a[i].x*b[j].x+a[i].y*b[j].y+a[i].z*b[j].z;
      int min=0,max=nbins;
      while (max>min+1) {
        int k=(min+max)/2;
        if (dot>=bounds[k]) max=k; else min=k;
      }
      if (dot>=bounds[min]) bins[min]++;
      else if (dot<bounds[max]) bins[max+1]++;
      else bins[max]++;
    }
}

int main(void) {
  enum { N1=5, N2=4, NB=4 };
  struct cartesian a[N1]={{1,0,0},{0,1,0},{0,0,1},{.5f,.5f,0},{-.5f,0,.5f}};
  struct cartesian b[N2]={{1,0,0},{0,-1,0},{.25f,.25f,.5f},{-1,0,0}};
  float bounds[NB+1]={1.0f,0.5f,0.0f,-0.5f,-1.0f};
  long long bins[NB+2]={3,1,4,1,5,9};
  long long expected[NB+2];
  for (int i=0;i<NB+2;++i) expected[i]=bins[i];
  reference(a,N1,b,N2,0,expected,NB,bounds);
  if (doCompute(a,N1,b,N2,0,bins,NB,bounds)!=0) return 2;
  for (int i=0;i<NB+2;++i) if (bins[i]!=expected[i]) {
    fprintf(stderr,"TPACF mismatch bin %d: %lld vs %lld\n",i,bins[i],expected[i]);
    return 1;
  }
  puts("parboil-tpacf-source: PASS");
  return 0;
}
