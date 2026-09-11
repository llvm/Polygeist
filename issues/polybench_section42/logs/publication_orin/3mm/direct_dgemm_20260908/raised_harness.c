#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define NI 800
#define NJ 900
#define NK 1000
#define NL 1100
#define NM 1200
extern void kernel_3mm(int,int,int,int,int,double[NI][NJ],double[NI][NK],
 double[NK][NJ],double[NJ][NL],double[NJ][NM],double[NM][NL],double[NI][NL]);
static double wall_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
static int run_once(int dump){
 double(*E)[NJ]=malloc(sizeof(double)*NI*NJ),(*A)[NK]=malloc(sizeof(double)*NI*NK),(*B)[NJ]=malloc(sizeof(double)*NK*NJ);
 double(*F)[NL]=malloc(sizeof(double)*NJ*NL),(*C)[NM]=malloc(sizeof(double)*NJ*NM),(*D)[NL]=malloc(sizeof(double)*NM*NL),(*G)[NL]=malloc(sizeof(double)*NI*NL);
 if(!E||!A||!B||!F||!C||!D||!G)return 2;
 for(int i=0;i<NI;i++)for(int j=0;j<NK;j++)A[i][j]=(double)((i*j+1)%NI)/(5*NI);
 for(int i=0;i<NK;i++)for(int j=0;j<NJ;j++)B[i][j]=(double)((i*(j+1)+2)%NJ)/(5*NJ);
 for(int i=0;i<NJ;i++)for(int j=0;j<NM;j++)C[i][j]=(double)(i*(j+3)%NL)/(5*NL);
 for(int i=0;i<NM;i++)for(int j=0;j<NL;j++)D[i][j]=(double)((i*(j+2)+2)%NK)/(5*NK);
 double b=wall_ms();kernel_3mm(NI,NJ,NK,NL,NM,E,A,B,F,C,D,G);printf("POLYGEIST_RAISED_GPU_E2E_MS %.9f\n",wall_ms()-b);
 if(dump){fprintf(stderr,"==BEGIN DUMP_ARRAYS==\nbegin dump: G\n");for(int i=0;i<NI;i++)for(int j=0;j<NL;j++){if((i*NI+j)%20==0)fprintf(stderr,"\n");fprintf(stderr,"%0.2lf ",G[i][j]);}fprintf(stderr,"\nend   dump: G\n==END   DUMP_ARRAYS==\n");}
 free(E);free(A);free(B);free(F);free(C);free(D);free(G);return 0;
}
int main(void){
#ifdef CORRECTNESS_RUN
 return run_once(1);
#else
 for(int i=0;i<10;i++){printf("POLYGEIST_BENCHMARK_ITERATION phase=%s index=%d total=10\n",i<5?"warmup":"sample",i<5?i:i-5);fflush(stdout);int s=run_once(0);if(s)return s;}return 0;
#endif
}
