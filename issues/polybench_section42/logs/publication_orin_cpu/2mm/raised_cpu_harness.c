#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define NI 800
#define NJ 900
#define NK 1100
#define NL 1200
extern void kernel_2mm(int,int,int,int,double,double,double[NI][NJ],
                       double[NI][NK],double[NK][NJ],double[NJ][NL],
                       double[NI][NL]);
static double wall_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
static int run_once(int dump){
 double(*tmp)[NJ]=malloc(sizeof(double)*NI*NJ); double(*A)[NK]=malloc(sizeof(double)*NI*NK);
 double(*B)[NJ]=malloc(sizeof(double)*NK*NJ); double(*C)[NL]=malloc(sizeof(double)*NJ*NL);
 double(*D)[NL]=malloc(sizeof(double)*NI*NL); if(!tmp||!A||!B||!C||!D)return 2;
 for(int i=0;i<NI;i++)for(int j=0;j<NK;j++)A[i][j]=(double)((i*j+1)%NI)/NI;
 for(int i=0;i<NK;i++)for(int j=0;j<NJ;j++)B[i][j]=(double)(i*(j+1)%NJ)/NJ;
 for(int i=0;i<NJ;i++)for(int j=0;j<NL;j++)C[i][j]=(double)((i*(j+3)+1)%NL)/NL;
 for(int i=0;i<NI;i++)for(int j=0;j<NL;j++)D[i][j]=(double)(i*(j+2)%NK)/NK;
 double b=wall_ms(); kernel_2mm(NI,NJ,NK,NL,1.5,1.2,tmp,A,B,C,D);
 printf("POLYGEIST_RAISED_GPU_E2E_MS %.9f\n",wall_ms()-b);
 if(dump){fprintf(stderr,"==BEGIN DUMP_ARRAYS==\nbegin dump: D\n");for(int i=0;i<NI;i++)for(int j=0;j<NL;j++){if((i*NI+j)%20==0)fprintf(stderr,"\n");fprintf(stderr,"%0.2lf ",D[i][j]);}fprintf(stderr,"\nend   dump: D\n==END   DUMP_ARRAYS==\n");}
 free(tmp);free(A);free(B);free(C);free(D);return 0;
}
int main(void){
#ifdef CORRECTNESS_RUN
 return run_once(1);
#else
 for(int i=0;i<10;i++){printf("POLYGEIST_BENCHMARK_ITERATION phase=%s index=%d total=10\n",i<5?"warmup":"sample",i<5?i:i-5);fflush(stdout);int s=run_once(0);if(s)return s;}return 0;
#endif
}
