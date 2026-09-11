#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define N 2000
extern void kernel_cholesky(int,double*);
static double ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.+t.tv_nsec/1e6;}
static int run(int dump){double(*A)[N]=malloc(sizeof(double)*N*N),(*B)[N]=malloc(sizeof(double)*N*N);if(!A||!B)return 2;for(int i=0;i<N;i++){for(int j=0;j<=i;j++)A[i][j]=(double)(-j%N)/N+1.;for(int j=i+1;j<N;j++)A[i][j]=0.;A[i][i]=1.;}for(int r=0;r<N;r++)for(int s=0;s<N;s++)B[r][s]=0.;for(int t=0;t<N;t++)for(int r=0;r<N;r++)for(int s=0;s<N;s++)B[r][s]+=A[r][t]*A[s][t];for(int r=0;r<N;r++)for(int s=0;s<N;s++)A[r][s]=B[r][s];double begin=ms();kernel_cholesky(N,&A[0][0]);printf("POLYGEIST_RAISED_CPU_MS %.9f\n",ms()-begin);if(dump){fprintf(stderr,"==BEGIN DUMP_ARRAYS==\nbegin dump: A\n");for(int i=0;i<N;i++)for(int j=0;j<=i;j++){if((i*N+j)%20==0)fprintf(stderr,"\n");fprintf(stderr,"%0.2lf ",A[i][j]);}fprintf(stderr,"\nend   dump: A\n==END   DUMP_ARRAYS==\n");}free(B);free(A);return 0;}
int main(void){
#ifdef CORRECTNESS_RUN
return run(1);
#else
for(int i=0;i<10;i++){printf("POLYGEIST_BENCHMARK_ITERATION phase=%s index=%d total=10\n",i<5?"warmup":"sample",i<5?i:i-5);fflush(stdout);int s=run(0);if(s)return s;}return 0;
#endif
}
