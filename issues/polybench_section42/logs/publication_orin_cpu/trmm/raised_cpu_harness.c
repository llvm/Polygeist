#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define M 1000
#define N 1200
extern void kernel_trmm(int,int,double,double[M][M],double[M][N]);
static double ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.+t.tv_nsec/1e6;}
static int run(int dump){double(*A)[M]=calloc(M*M,sizeof(double)),(*B)[N]=malloc(sizeof(double)*M*N);if(!A||!B)return 2;for(int i=0;i<M;i++){for(int j=0;j<i;j++)A[i][j]=(double)((i+j)%M)/M;A[i][i]=1;for(int j=0;j<N;j++)B[i][j]=(double)((N+(i-j))%N)/N;}double s=ms();kernel_trmm(M,N,1.5,A,B);printf("POLYGEIST_RAISED_CPU_MS %.9f\n",ms()-s);if(dump){fprintf(stderr,"==BEGIN DUMP_ARRAYS==\nbegin dump: B\n");for(int i=0;i<M;i++)for(int j=0;j<N;j++){if((i*M+j)%20==0)fprintf(stderr,"\n");fprintf(stderr,"%0.2lf ",B[i][j]);}fprintf(stderr,"\nend   dump: B\n==END   DUMP_ARRAYS==\n");}free(A);free(B);return 0;}
int main(void){
#ifdef CORRECTNESS_RUN
return run(1);
#else
for(int i=0;i<10;i++){printf("POLYGEIST_BENCHMARK_ITERATION phase=%s index=%d total=10\n",i<5?"warmup":"sample",i<5?i:i-5);fflush(stdout);int s=run(0);if(s)return s;}return 0;
#endif
}
