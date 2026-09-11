#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifndef N
#define N 256
#endif
#ifndef BENCH_ITERS
#define BENCH_ITERS 5
#endif
extern void FUNCTION(uint8_t *, float, int, uint8_t *);
#define S1(x) #x
#define S(x) S1(x)
static double seconds(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(void){uint8_t *in=malloc(N),*out=malloc(N);float scale=.037f;int zero=113;if(!in||!out)return 2;for(int i=0;i<N;i++)in[i]=(uint8_t)((i*73+19)&255);FUNCTION(in,scale,zero,out);for(int i=0;i<N;i++){float v=((int)in[i]-zero)*scale;int q=(int)(v/scale)+zero;if(q<0)q=0;if(q>255)q=255;if(out[i]!=(uint8_t)q){fprintf(stderr,"FAIL i=%d got=%u expected=%d\n",i,out[i],q);return 1;}}double b=seconds();for(int it=0;it<BENCH_ITERS;it++)FUNCTION(in,scale,zero,out);printf("RESULT function=%s elements=%d warm_us=%.6f correctness=PASS\n",S(FUNCTION),N,(seconds()-b)*1e6/BENCH_ITERS);return 0;}
