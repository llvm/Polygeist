#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 1300
extern void kernel_gesummv(int,double,double,double a[N][N],double b[N][N],double tmp[N],double x[N],double y[N]);
static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
static void initialize(double a[N][N],double b[N][N],double x[N]){for(int i=0;i<N;++i){x[i]=(double)(i%N)/N;for(int j=0;j<N;++j){a[i][j]=(double)((i*j+1)%N)/N;b[i][j]=(double)((i*j+2)%N)/N;}}}
int main(void){double(*a)[N]=malloc(sizeof(double[N][N])),(*b)[N]=malloc(sizeof(double[N][N]));double*tmp=malloc(sizeof(double[N])),*x=malloc(sizeof(double[N])),*y=malloc(sizeof(double[N]));if(!a||!b||!tmp||!x||!y)return 2;initialize(a,b,x);
#ifdef POLYGEIST_CPU_TIMING
for(int i=0;i<10;++i){double t=now_ms();kernel_gesummv(N,1.5,1.2,a,b,tmp,x,y);double e=now_ms()-t;printf("CPU_SAMPLE phase=%s index=%d runtime_ms=%.6f\n",i<5?"warmup":"sample",i<5?i:i-5,e);fflush(stdout);}
#else
kernel_gesummv(N,1.5,1.2,a,b,tmp,x,y);fprintf(stderr,"==BEGIN DUMP_ARRAYS==\nbegin dump: y");for(int i=0;i<N;++i){if(i%20==0)fprintf(stderr,"\n");fprintf(stderr,"%0.2lf ",y[i]);}fprintf(stderr,"\nend   dump: y\n==END   DUMP_ARRAYS==\n");
#endif
free(y);free(x);free(tmp);free(b);free(a);return 0;}
