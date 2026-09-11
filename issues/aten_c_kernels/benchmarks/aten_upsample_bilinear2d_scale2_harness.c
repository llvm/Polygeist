#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef B
#define B 2
#define C 3
#define H 4
#define W 4
#endif
extern void FUNCTION(float *, float *);
int main(void){size_t outer=(size_t)B*C;float*in=malloc(outer*H*W*4),*out=malloc(outer*4*H*W*4);for(size_t i=0;i<outer*H*W;i++)in[i]=(i%101)*.125f;for(int run=0;run<4;run++)FUNCTION(in,out);for(size_t b=0;b<outer;b++)for(int y=0;y<2*H;y++)for(int x=0;x<2*W;x++){int y0=y/2,x0=x/2,y1=y0+1<H?y0+1:y0,x1=x0+1<W?x0+1:x0;float wy=(y%2)*.5f,wx=(x%2)*.5f,*p=in+b*H*W;float want=(1-wy)*((1-wx)*p[y0*W+x0]+wx*p[y0*W+x1])+wy*((1-wx)*p[y1*W+x0]+wx*p[y1*W+x1]),got=out[(b*2*H+y)*2*W+x];if(fabsf(got-want)>2e-5f*fmaxf(1,fabsf(want))){fprintf(stderr,"FAIL got=%g want=%g\n",got,want);return 1;}}puts("RESULT correctness=PASS");return 0;}
