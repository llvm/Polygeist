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
int main(void) {
  size_t outer=(size_t)B*C, ni=outer*I0*I1*I2, no=outer*O0*O1*O2;
  float *go=malloc(no*4),*gi=malloc(ni*4),*ref=calloc(ni,4);
  for(size_t i=0;i<no;i++)go[i]=((int)(i%17)-8)*.0625f;
  for(size_t b=0;b<outer;b++)for(int z=0;z<O0;z++)for(int y=0;y<O1;y++)for(int x=0;x<O2;x++){
    float sz=(z+.5f)*I0/O0-.5f,sy=(y+.5f)*I1/O1-.5f,sx=(x+.5f)*I2/O2-.5f;
    if(sz<0)sz=0;if(sy<0)sy=0;if(sx<0)sx=0;
    int z0=(int)sz,y0=(int)sy,x0=(int)sx,z1=z0+1<I0?z0+1:z0,y1=y0+1<I1?y0+1:y0,x1=x0+1<I2?x0+1:x0;
    float wz=sz-z0,wy=sy-y0,wx=sx-x0,v=go[((b*O0+z)*O1+y)*O2+x];float*p=ref+b*I0*I1*I2;
    p[(z0*I1+y0)*I2+x0]+=v*(1-wz)*(1-wy)*(1-wx);p[(z0*I1+y0)*I2+x1]+=v*(1-wz)*(1-wy)*wx;
    p[(z0*I1+y1)*I2+x0]+=v*(1-wz)*wy*(1-wx);p[(z0*I1+y1)*I2+x1]+=v*(1-wz)*wy*wx;
    p[(z1*I1+y0)*I2+x0]+=v*wz*(1-wy)*(1-wx);p[(z1*I1+y0)*I2+x1]+=v*wz*(1-wy)*wx;
    p[(z1*I1+y1)*I2+x0]+=v*wz*wy*(1-wx);p[(z1*I1+y1)*I2+x1]+=v*wz*wy*wx;
  }
  for(int run=0;run<4;run++)FUNCTION(go,gi);
  for(size_t i=0;i<ni;i++)if(fabsf(gi[i]-ref[i])>5e-5f*fmaxf(1,fabsf(ref[i]))){fprintf(stderr,"FAIL i=%zu got=%g want=%g\n",i,gi[i],ref[i]);return 1;}
  puts("RESULT correctness=PASS");return 0;
}
