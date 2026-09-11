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
extern void FUNCTION(float *, float *);
int main(void) {
  size_t outer = (size_t)B * C;
  float *go = malloc(outer * O0 * O1 * sizeof(float));
  float *gi = malloc(outer * I0 * I1 * sizeof(float));
  float *ref = calloc(outer * I0 * I1, sizeof(float));
  for (size_t i = 0; i < outer * O0 * O1; ++i)
    go[i] = ((int)(i % 17) - 8) * .0625f;
  for (size_t b = 0; b < outer; ++b)
    for (int y = 0; y < O0; ++y)
      for (int x = 0; x < O1; ++x) {
        float sy=(y+.5f)*I0/O0-.5f,sx=(x+.5f)*I1/O1-.5f;
        if(sy<0)sy=0;if(sx<0)sx=0;
        int y0=(int)sy,x0=(int)sx,y1=y0+1<I0?y0+1:y0,x1=x0+1<I1?x0+1:x0;
        float wy=sy-y0,wx=sx-x0,v=go[(b*O0+y)*O1+x];float*p=ref+b*I0*I1;
        p[y0*I1+x0]+=v*(1-wy)*(1-wx);p[y0*I1+x1]+=v*(1-wy)*wx;
        p[y1*I1+x0]+=v*wy*(1-wx);p[y1*I1+x1]+=v*wy*wx;
      }
  for(int run=0;run<4;run++) FUNCTION(go, gi);
  for(size_t i=0;i<outer*I0*I1;i++)if(fabsf(gi[i]-ref[i])>3e-5f*fmaxf(1,fabsf(ref[i]))){fprintf(stderr,"FAIL i=%zu got=%g want=%g\n",i,gi[i],ref[i]);return 1;}
  puts("RESULT correctness=PASS");return 0;
}
