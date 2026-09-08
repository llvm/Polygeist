#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define NR 150
#define NQ 140
#define NP 160
extern void kernel_doitgen(int,int,int,double[NR][NQ][NP],double[NP][NP],double[NP]);
static double ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.+t.tv_nsec/1e6;}
static int run(int dump){double(*A)[NQ][NP]=malloc(sizeof(double)*NR*NQ*NP),(*C4)[NP]=malloc(sizeof(double)*NP*NP),*sum=malloc(sizeof(double)*NP);if(!A||!C4||!sum)return 2;for(int i=0;i<NR;i++)for(int j=0;j<NQ;j++)for(int k=0;k<NP;k++)A[i][j][k]=(double)((i*j+k)%NP)/NP;for(int i=0;i<NP;i++)for(int j=0;j<NP;j++)C4[i][j]=(double)(i*j%NP)/NP;double s=ms();kernel_doitgen(NR,NQ,NP,A,C4,sum);printf("POLYGEIST_RAISED_CPU_MS %.9f\n",ms()-s);if(dump){fprintf(stderr,"==BEGIN DUMP_ARRAYS==\nbegin dump: A\n");for(int i=0;i<NR;i++)for(int j=0;j<NQ;j++)for(int k=0;k<NP;k++){if((i*NQ*NP+j*NP+k)%20==0)fprintf(stderr,"\n");fprintf(stderr,"%0.2lf ",A[i][j][k]);}fprintf(stderr,"\nend   dump: A\n==END   DUMP_ARRAYS==\n");}free(A);free(C4);free(sum);return 0;}
int main(void){
#ifdef CORRECTNESS_RUN
return run(1);
#else
for(int i=0;i<10;i++){printf("POLYGEIST_BENCHMARK_ITERATION phase=%s index=%d total=10\n",i<5?"warmup":"sample",i<5?i:i-5);fflush(stdout);int s=run(0);if(s)return s;}return 0;
#endif
}
