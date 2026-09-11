#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define NI 1000
#define NJ 1100
#define NK 1200
extern void kernel_gemm(int,int,int,double,double,double*,double*,double*);
static double ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.+t.tv_nsec/1e6;}
static int run(int dump){double*C=malloc(sizeof(double)*NI*NJ),*A=malloc(sizeof(double)*NI*NK),*B=malloc(sizeof(double)*NK*NJ);if(!A||!B||!C)return 2;for(int i=0;i<NI;i++)for(int j=0;j<NJ;j++)C[i*NJ+j]=(double)((i*j+1)%NI)/NI;for(int i=0;i<NI;i++)for(int j=0;j<NK;j++)A[i*NK+j]=(double)(i*(j+1)%NK)/NK;for(int i=0;i<NK;i++)for(int j=0;j<NJ;j++)B[i*NJ+j]=(double)(i*(j+2)%NJ)/NJ;double s=ms();kernel_gemm(NI,NJ,NK,1.5,1.2,C,A,B);printf("POLYGEIST_RAISED_CPU_MS %.9f\n",ms()-s);if(dump){fprintf(stderr,"==BEGIN DUMP_ARRAYS==\nbegin dump: C\n");for(int i=0;i<NI;i++)for(int j=0;j<NJ;j++){if((i*NI+j)%20==0)fprintf(stderr,"\n");fprintf(stderr,"%0.2lf ",C[i*NJ+j]);}fprintf(stderr,"\nend   dump: C\n==END   DUMP_ARRAYS==\n");}free(A);free(B);free(C);return 0;}
int main(void){
#ifdef CORRECTNESS_RUN
return run(1);
#else
for(int i=0;i<10;i++){printf("POLYGEIST_BENCHMARK_ITERATION phase=%s index=%d total=10\n",i<5?"warmup":"sample",i<5?i:i-5);fflush(stdout);int s=run(0);if(s)return s;}return 0;
#endif
}
