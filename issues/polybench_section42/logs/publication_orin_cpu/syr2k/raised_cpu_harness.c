#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 1200
#define M 1000
extern void kernel_syr2k(int,int,double,double,double[N][N],double[N][M],double[N][M]);
static double wall_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
static int run_once(int dump){double(*C)[N]=malloc(sizeof(double)*N*N),(*A)[M]=malloc(sizeof(double)*N*M),(*B)[M]=malloc(sizeof(double)*N*M);if(!C||!A||!B)return 2;for(int i=0;i<N;i++)for(int j=0;j<M;j++){A[i][j]=(double)((i*j+1)%N)/N;B[i][j]=(double)((i*j+2)%M)/M;}for(int i=0;i<N;i++)for(int j=0;j<N;j++)C[i][j]=(double)((i*j+3)%N)/M;double b=wall_ms();kernel_syr2k(N,M,1.5,1.2,C,A,B);printf("POLYGEIST_RAISED_GPU_E2E_MS %.9f\n",wall_ms()-b);if(dump){fprintf(stderr,"==BEGIN DUMP_ARRAYS==\nbegin dump: C\n");for(int i=0;i<N;i++)for(int j=0;j<N;j++){if((i*N+j)%20==0)fprintf(stderr,"\n");fprintf(stderr,"%0.2lf ",C[i][j]);}fprintf(stderr,"\nend   dump: C\n==END   DUMP_ARRAYS==\n");}free(C);free(A);free(B);return 0;}
int main(void){
#ifdef CORRECTNESS_RUN
return run_once(1);
#else
for(int i=0;i<10;i++){printf("POLYGEIST_BENCHMARK_ITERATION phase=%s index=%d total=10\n",i<5?"warmup":"sample",i<5?i:i-5);fflush(stdout);int s=run_once(0);if(s)return s;}return 0;
#endif
}
