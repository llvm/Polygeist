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
extern void FUNCTION(float*,float*,float*,float*,float*);
int main(void){size_t ni=(size_t)B*C*IH*IW,ng=(size_t)B*OH*OW*2,no=(size_t)B*C*OH*OW;float*in=malloc(ni*4),*grid=malloc(ng*4),*grad=malloc(no*4),*dx=malloc(ni*4),*dg=malloc(ng*4),*rdx=calloc(ni,4),*rdg=calloc(ng,4);for(size_t i=0;i<ni;i++)in[i]=(i%101)*.03125f;for(size_t i=0;i<ng;i++)grid[i]=((int)(i%19)-9)/10.f;for(size_t i=0;i<no;i++)grad[i]=((int)(i%17)-8)*.03125f;for(int b=0;b<B;b++)for(int y=0;y<OH;y++)for(int x=0;x<OW;x++){int q=(b*OH+y)*OW+x;float fx=(grid[q*2]+1)*.5f*(IW-1),fy=(grid[q*2+1]+1)*.5f*(IH-1);int x0=(int)fx,y0=(int)fy,x1=x0+1,y1=y0+1;float wx=fx-x0,wy=fy-y0,gx=0,gy=0;for(int c=0;c<C;c++){float g=grad[((b*C+c)*OH+y)*OW+x],v00=0,v01=0,v10=0,v11=0;if(x0>=0&&x0<IW&&y0>=0&&y0<IH){v00=in[((b*C+c)*IH+y0)*IW+x0];rdx[((b*C+c)*IH+y0)*IW+x0]+=g*(1-wx)*(1-wy);}if(x1>=0&&x1<IW&&y0>=0&&y0<IH){v01=in[((b*C+c)*IH+y0)*IW+x1];rdx[((b*C+c)*IH+y0)*IW+x1]+=g*wx*(1-wy);}if(x0>=0&&x0<IW&&y1>=0&&y1<IH){v10=in[((b*C+c)*IH+y1)*IW+x0];rdx[((b*C+c)*IH+y1)*IW+x0]+=g*(1-wx)*wy;}if(x1>=0&&x1<IW&&y1>=0&&y1<IH){v11=in[((b*C+c)*IH+y1)*IW+x1];rdx[((b*C+c)*IH+y1)*IW+x1]+=g*wx*wy;}gx+=g*((v01-v00)*(1-wy)+(v11-v10)*wy);gy+=g*((v10-v00)*(1-wx)+(v11-v01)*wx);}rdg[q*2]=gx*.5f*(IW-1);rdg[q*2+1]=gy*.5f*(IH-1);}for(int run=0;run<4;run++)FUNCTION(in,grid,grad,dx,dg);for(size_t i=0;i<ni;i++)if(fabsf(dx[i]-rdx[i])>8e-5f*fmaxf(1,fabsf(rdx[i]))){fprintf(stderr,"DX FAIL %zu\n",i);return 1;}for(size_t i=0;i<ng;i++)if(fabsf(dg[i]-rdg[i])>2e-4f*fmaxf(1,fabsf(rdg[i]))){fprintf(stderr,"DG FAIL %zu got=%g want=%g\n",i,dg[i],rdg[i]);return 1;}puts("RESULT correctness=PASS");return 0;}
