#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define M 1000
#define N 1200
extern void kernel_symm(int,int,double,double,double[M][N],double[M][M],double[M][N]);
static double ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.+t.tv_nsec/1e6;}
static int run(int dump){double(*C)[N]=malloc(sizeof(double)*M*N),(*A)[M]=malloc(sizeof(double)*M*M),(*B)[N]=malloc(sizeof(double)*M*N);if(!A||!B||!C)return 2;for(int i=0;i<M;i++)for(int j=0;j<N;j++){C[i][j]=(double)((i+j)%100)/M;B[i][j]=(double)((N+i-j)%100)/M;}for(int i=0;i<M;i++){for(int j=0;j<=i;j++)A[i][j]=(double)((i+j)%100)/M;for(int j=i+1;j<M;j++)A[i][j]=-999;}double s=ms();kernel_symm(M,N,1.5,1.2,C,A,B);printf("POLYGEIST_RAISED_GPU_E2E_MS %.9f\n",ms()-s);if(dump){fprintf(stderr,"==BEGIN DUMP_ARRAYS==\nbegin dump: C\n");for(int i=0;i<M;i++)for(int j=0;j<N;j++){if((i*M+j)%20==0)fprintf(stderr,"\n");fprintf(stderr,"%0.2lf ",C[i][j]);}fprintf(stderr,"\nend   dump: C\n==END   DUMP_ARRAYS==\n");}free(A);free(B);free(C);return 0;}
int main(void){
#ifdef CORRECTNESS_RUN
return run(1);
#else
for(int i=0;i<10;i++){printf("POLYGEIST_BENCHMARK_ITERATION phase=%s index=%d total=10\n",i<5?"warmup":"sample",i<5?i:i-5);fflush(stdout);int s=run(0);if(s)return s;}return 0;
#endif
}
