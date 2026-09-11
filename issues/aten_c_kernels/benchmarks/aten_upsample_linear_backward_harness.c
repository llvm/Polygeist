#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef B
#define B 1
#define C 2
#define I0 4
#define O0 7
#endif
extern void FUNCTION(float*,float*);
int main(void){size_t outer=(size_t)B*C;float *go=malloc(outer*O0*4),*gi=malloc(outer*I0*4),*ref=calloc(outer*I0,4);for(size_t i=0;i<outer*O0;i++)go[i]=((int)(i%17)-8)*.0625f;for(size_t b=0;b<outer;b++)for(int o=0;o<O0;o++){float s=(o+.5f)*I0/O0-.5f;if(s<0)s=0;int lo=(int)s,hi=lo+1<I0?lo+1:lo;float wh=s-lo;ref[b*I0+lo]+=go[b*O0+o]*(1-wh);ref[b*I0+hi]+=go[b*O0+o]*wh;}for(int run=0;run<4;run++)FUNCTION(go,gi);for(size_t i=0;i<outer*I0;i++)if(fabsf(gi[i]-ref[i])>2e-5f*fmaxf(1,fabsf(ref[i]))){fprintf(stderr,"FAIL i=%zu got=%g want=%g\n",i,gi[i],ref[i]);return 1;}puts("RESULT correctness=PASS");return 0;}
