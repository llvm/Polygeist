#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 2000
extern void kernel_mvt(int,double[N],double[N],double[N],double[N],double[N][N]);
static double wall_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
static int run_once(int dump){double(*A)[N]=malloc(sizeof(double)*N*N),*x1=malloc(sizeof(double)*N),*x2=malloc(sizeof(double)*N),*y1=malloc(sizeof(double)*N),*y2=malloc(sizeof(double)*N);if(!A||!x1||!x2||!y1||!y2)return 2;
 for(int i=0;i<N;i++){x1[i]=(double)(i%N)/N;x2[i]=(double)((i+1)%N)/N;y1[i]=(double)((i+3)%N)/N;y2[i]=(double)((i+4)%N)/N;for(int j=0;j<N;j++)A[i][j]=(double)(i*j%N)/N;}
 double b=wall_ms();kernel_mvt(N,x1,x2,y1,y2,A);printf("POLYGEIST_RAISED_GPU_E2E_MS %.9f\n",wall_ms()-b);
 if(dump){fprintf(stderr,"==BEGIN DUMP_ARRAYS==\nbegin dump: x1\n");for(int i=0;i<N;i++){if(i%20==0)fprintf(stderr,"\n");fprintf(stderr,"%0.2lf ",x1[i]);}fprintf(stderr,"\nend   dump: x1\nbegin dump: x2\n");for(int i=0;i<N;i++){if(i%20==0)fprintf(stderr,"\n");fprintf(stderr,"%0.2lf ",x2[i]);}fprintf(stderr,"\nend   dump: x2\n==END   DUMP_ARRAYS==\n");}free(A);free(x1);free(x2);free(y1);free(y2);return 0;}
int main(void){
#ifdef CORRECTNESS_RUN
 return run_once(1);
#else
 for(int i=0;i<10;i++){printf("POLYGEIST_BENCHMARK_ITERATION phase=%s index=%d total=10\n",i<5?"warmup":"sample",i<5?i:i-5);fflush(stdout);int s=run_once(0);if(s)return s;}return 0;
#endif
}
