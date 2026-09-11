#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef B
#define B 1
#define C 2
#define I0 4
#define I1 5
#define O0 7
#define O1 8
#endif
extern void FUNCTION(float*,float*);
int main(void){size_t outer=(size_t)B*C;float *in=malloc(outer*I0*I1*4),*out=malloc(outer*O0*O1*4);for(size_t i=0;i<outer*I0*I1;i++)in[i]=(i%101)*.125f;for(int run=0;run<4;run++)FUNCTION(in,out);for(size_t b=0;b<outer;b++)for(int y=0;y<O0;y++)for(int x=0;x<O1;x++){float sy=(y+.5f)*I0/O0-.5f,sx=(x+.5f)*I1/O1-.5f;if(sy<0)sy=0;if(sx<0)sx=0;int y0=(int)sy,x0=(int)sx,y1=y0+1<I0?y0+1:y0,x1=x0+1<I1?x0+1:x0;float wy=sy-y0,wx=sx-x0;float*p=in+b*I0*I1;float want=p[y0*I1+x0]*(1-wy)*(1-wx)+p[y0*I1+x1]*(1-wy)*wx+p[y1*I1+x0]*wy*(1-wx)+p[y1*I1+x1]*wy*wx,got=out[(b*O0+y)*O1+x];if(fabsf(got-want)>2e-5f*fmaxf(1,fabsf(want))){fprintf(stderr,"FAIL %zu %d %d got=%g want=%g\n",b,y,x,got,want);return 1;}}puts("RESULT correctness=PASS");return 0;}
