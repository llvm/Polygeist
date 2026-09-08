#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 1900
#define N 2100
extern void kernel_bicg(int, int, double a[N][M], double s[M], double q[N],
                        double p[M], double r[N]);
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1000.0+t.tv_nsec/1e6; }
static void initialize(double a[N][M], double r[N], double p[M]) {
  for (int i=0;i<M;++i) p[i]=(double)(i%M)/M;
  for (int i=0;i<N;++i) { r[i]=(double)(i%N)/N; for(int j=0;j<M;++j) a[i][j]=(double)(i*(j+1)%N)/N; }
}
static void dump(const char *name, double *x, int n) {
  fprintf(stderr,"begin dump: %s",name); for(int i=0;i<n;++i){if(i%20==0)fprintf(stderr,"\n");fprintf(stderr,"%0.2lf ",x[i]);} fprintf(stderr,"\nend   dump: %s\n",name);
}
int main(void) {
  double(*a)[M]=malloc(sizeof(double[N][M])); double *s=malloc(sizeof(double[M])),*q=malloc(sizeof(double[N])),*p=malloc(sizeof(double[M])),*r=malloc(sizeof(double[N]));
  if(!a||!s||!q||!p||!r)return 2; initialize(a,r,p);
#ifdef POLYGEIST_CPU_TIMING
  for(int i=0;i<10;++i){double t=now_ms();kernel_bicg(M,N,a,s,q,p,r);double e=now_ms()-t;printf("CPU_SAMPLE phase=%s index=%d runtime_ms=%.6f\n",i<5?"warmup":"sample",i<5?i:i-5,e);fflush(stdout);}
#else
  kernel_bicg(M,N,a,s,q,p,r); fprintf(stderr,"==BEGIN DUMP_ARRAYS==\n");dump("s",s,M);dump("q",q,N);fprintf(stderr,"==END   DUMP_ARRAYS==\n");
#endif
  free(r);free(p);free(q);free(s);free(a);return 0;
}
