#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifndef RANK
#define RANK 2
#endif
#ifndef EXACT
#define EXACT 0
#endif
#ifndef BENCH_ITERS
#define BENCH_ITERS 5
#endif
#ifndef B
#define B 1
#define C 2
#define I0 4
#define O0 7
#endif
#ifndef I1
#define I1 1
#define O1 1
#endif
#ifndef I2
#define I2 1
#define O2 1
#endif
extern void FUNCTION(float *, float *);
#define S1(x) #x
#define S(x) S1(x)
static double seconds(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
int main(void) {
  int ids[3]={I0,I1,I2}, ods[3]={O0,O1,O2};
  size_t outer=(size_t)B*C, si=1, so=1;
  for(int d=0;d<RANK;d++){si*=ids[d];so*=ods[d];}
  float *go=malloc(outer*so*4), *gi=malloc(outer*si*4), *ref=calloc(outer*si,4);
  if(!go||!gi||!ref)return 2;
  for(size_t i=0;i<outer*so;i++)go[i]=(float)((int)(i%31)-15)*.03125f;
  for(size_t b=0;b<outer;b++)for(size_t linear=0;linear<so;linear++){
    size_t rem=linear,dst=0,stride=1;
    for(int d=RANK-1;d>=0;d--){int c=rem%ods[d];rem/=ods[d];int m=EXACT?((2*c+1)*ids[d])/(2*ods[d]):c*ids[d]/ods[d];if(m>=ids[d])m=ids[d]-1;dst+=m*stride;stride*=ids[d];}
    ref[b*si+dst]+=go[b*so+linear];
  }
  FUNCTION(go,gi);
  for(size_t i=0;i<outer*si;i++)if(fabsf(gi[i]-ref[i])>2e-5f*fmaxf(1,fabsf(ref[i]))){fprintf(stderr,"FAIL i=%zu got=%g expected=%g\n",i,gi[i],ref[i]);return 1;}
  double begin=seconds();for(int it=0;it<BENCH_ITERS;it++)FUNCTION(go,gi);
  printf("RESULT function=%s output_grad_elements=%zu input_grad_elements=%zu warm_us=%.6f correctness=PASS\n",S(FUNCTION),outer*so,outer*si,(seconds()-begin)*1e6/BENCH_ITERS);
  return 0;
}
