#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 2000
extern void kernel_trisolv(int,double[N][N],double[N],double[N]);
static double ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.+t.tv_nsec/1e6;}
static int run(int dump){double(*L)[N]=calloc(N*N,sizeof(double)),*x=malloc(sizeof(double)*N),*b=malloc(sizeof(double)*N);if(!L||!x||!b)return 2;for(int i=0;i<N;i++){x[i]=-999;b[i]=i;for(int j=0;j<=i;j++)L[i][j]=(double)(i+N-j+1)*2/N;}double s=ms();kernel_trisolv(N,L,x,b);printf("POLYGEIST_RAISED_CPU_MS %.9f\n",ms()-s);if(dump){fprintf(stderr,"==BEGIN DUMP_ARRAYS==\nbegin dump: x");for(int i=0;i<N;i++){fprintf(stderr,"%0.2lf ",x[i]);if(i%20==0)fprintf(stderr,"\n");}fprintf(stderr,"\nend   dump: x\n==END   DUMP_ARRAYS==\n");}free(L);free(x);free(b);return 0;}
int main(void){
#ifdef CORRECTNESS_RUN
return run(1);
#else
for(int i=0;i<10;i++){printf("POLYGEIST_BENCHMARK_ITERATION phase=%s index=%d total=10\n",i<5?"warmup":"sample",i<5?i:i-5);fflush(stdout);int s=run(0);if(s)return s;}return 0;
#endif
}
