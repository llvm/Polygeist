#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef B
#define B 1
#define C 3
#define IH 8
#define IW 8
#define OH 6
#define OW 6
#endif
extern void FUNCTION(float *, float *, float *);
int main(void) {
  size_t ni=(size_t)B*C*IH*IW, ng=(size_t)B*OH*OW*2, no=(size_t)B*C*OH*OW;
  float *in=malloc(ni*4), *grid=malloc(ng*4), *out=malloc(no*4);
  if(!in||!grid||!out) return 2;
  for(size_t i=0;i<ni;i++) in[i]=(i%97)*0.125f;
  for(int y=0;y<OH;y++) for(int x=0;x<OW;x++) {
    size_t p=((size_t)y*OW+x)*2;
    grid[p]=OH==1?0.0f:-1.0f+2.0f*x/(OW-1);
    grid[p+1]=OW==1?0.0f:-1.0f+2.0f*y/(OH-1);
  }
  FUNCTION(in,grid,out);
  for(int c=0;c<C;c++) for(int y=0;y<OH;y++) for(int x=0;x<OW;x++) {
    float fx=(grid[(y*OW+x)*2]+1)*.5f*(IW-1), fy=(grid[(y*OW+x)*2+1]+1)*.5f*(IH-1);
    int x0=(int)fx,y0=(int)fy,x1=x0+1,y1=y0+1; float wx=fx-x0,wy=fy-y0,v=0;
    #define ADD(XX,YY,W) do { if((XX)>=0&&(XX)<IW&&(YY)>=0&&(YY)<IH) v+=(W)*in[((size_t)c*IH+(YY))*IW+(XX)]; } while(0)
    ADD(x0,y0,(1-wx)*(1-wy)); ADD(x1,y0,wx*(1-wy)); ADD(x0,y1,(1-wx)*wy); ADD(x1,y1,wx*wy);
    float got=out[((size_t)c*OH+y)*OW+x]; if(fabsf(got-v)>1e-5f*fmaxf(1,fabsf(v))) {fprintf(stderr,"FAIL c=%d y=%d x=%d got=%g want=%g\n",c,y,x,got,v);return 1;}
  }
  printf("RESULT output_elements=%zu correctness=PASS\n",no); return 0;
}
