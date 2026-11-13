#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define M 1200
#define N 1400
extern void kernel_covariance(int,int,double,double[N][M],double[M][M],double[M]);
static double wall_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
static int run_once(int dump){double(*data)[M]=malloc(sizeof(double)*N*M),(*cov)[M]=malloc(sizeof(double)*M*M),*mean=malloc(sizeof(double)*M);if(!data||!cov||!mean)return 2;for(int i=0;i<N;i++)for(int j=0;j<M;j++)data[i][j]=(double)(i*j)/M;
 double b=wall_ms();kernel_covariance(M,N,(double)N,data,cov,mean);printf("POLYGEIST_RAISED_GPU_E2E_MS %.9f\n",wall_ms()-b);
 if(dump){fprintf(stderr,"==BEGIN DUMP_ARRAYS==\nbegin dump: cov\n");for(int i=0;i<M;i++)for(int j=0;j<M;j++){if((i*M+j)%20==0)fprintf(stderr,"\n");fprintf(stderr,"%0.2lf ",cov[i][j]);}fprintf(stderr,"\nend   dump: cov\n==END   DUMP_ARRAYS==\n");}free(data);free(cov);free(mean);return 0;}
int main(void){
#ifdef CORRECTNESS_RUN
 return run_once(1);
#else
 for(int i=0;i<10;i++){printf("POLYGEIST_BENCHMARK_ITERATION phase=%s index=%d total=10\n",i<5?"warmup":"sample",i<5?i:i-5);fflush(stdout);int s=run_once(0);if(s)return s;}return 0;
#endif
}
