#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef B
#define B 1
#define C 2
#define I0 4
#define I1 5
#define I2 6
#define O0 7
#define O1 8
#define O2 9
#endif
extern void FUNCTION(float *, float *);
int main(void){size_t outer=(size_t)B*C;float*in=malloc(outer*I0*I1*I2*4),*out=malloc(outer*O0*O1*O2*4);for(size_t i=0;i<outer*I0*I1*I2;i++)in[i]=(i%101)*.125f;for(int run=0;run<4;run++)FUNCTION(in,out);for(size_t b=0;b<outer;b++)for(int z=0;z<O0;z++)for(int y=0;y<O1;y++)for(int x=0;x<O2;x++){float sz=(z+.5f)*I0/O0-.5f,sy=(y+.5f)*I1/O1-.5f,sx=(x+.5f)*I2/O2-.5f;if(sz<0)sz=0;if(sy<0)sy=0;if(sx<0)sx=0;int z0=(int)sz,y0=(int)sy,x0=(int)sx,z1=z0+1<I0?z0+1:z0,y1=y0+1<I1?y0+1:y0,x1=x0+1<I2?x0+1:x0;float wz=sz-z0,wy=sy-y0,wx=sx-x0;float*p=in+b*I0*I1*I2;float want=p[(z0*I1+y0)*I2+x0]*(1-wz)*(1-wy)*(1-wx)+p[(z0*I1+y0)*I2+x1]*(1-wz)*(1-wy)*wx+p[(z0*I1+y1)*I2+x0]*(1-wz)*wy*(1-wx)+p[(z0*I1+y1)*I2+x1]*(1-wz)*wy*wx+p[(z1*I1+y0)*I2+x0]*wz*(1-wy)*(1-wx)+p[(z1*I1+y0)*I2+x1]*wz*(1-wy)*wx+p[(z1*I1+y1)*I2+x0]*wz*wy*(1-wx)+p[(z1*I1+y1)*I2+x1]*wz*wy*wx,got=out[((b*O0+z)*O1+y)*O2+x];if(fabsf(got-want)>3e-5f*fmaxf(1,fabsf(want))){fprintf(stderr,"FAIL got=%g want=%g\n",got,want);return 1;}}puts("RESULT correctness=PASS");return 0;}
