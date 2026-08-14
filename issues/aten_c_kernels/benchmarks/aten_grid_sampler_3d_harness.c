#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef B
#define B 1
#define C 2
#define ID 6
#define IH 7
#define IW 8
#define OD 4
#define OH 5
#define OW 6
#endif
extern void FUNCTION(float *, float *, float *);
int main(void){size_t ni=(size_t)B*C*ID*IH*IW,ng=(size_t)B*OD*OH*OW*3,no=(size_t)B*C*OD*OH*OW;float*in=malloc(ni*4),*g=malloc(ng*4),*out=malloc(no*4),*ref=calloc(no,4);for(size_t i=0;i<ni;i++)in[i]=(i%101)*.03125f;for(size_t i=0;i<ng;i++)g[i]=((int)(i%19)-9)/10.0f;for(int b=0;b<B;b++)for(int z=0;z<OD;z++)for(int y=0;y<OH;y++)for(int w=0;w<OW;w++){int q=((b*OD+z)*OH+y)*OW+w;float fx=(g[q*3]+1)*.5f*(IW-1),fy=(g[q*3+1]+1)*.5f*(IH-1),fz=(g[q*3+2]+1)*.5f*(ID-1);int x0=(int)fx,y0=(int)fy,z0=(int)fz;float ax=fx-x0,ay=fy-y0,az=fz-z0;for(int c=0;c<C;c++){float v=0;for(int dz=0;dz<2;dz++)for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++){int iz=z0+dz,iy=y0+dy,ix=x0+dx;if(iz>=0&&iz<ID&&iy>=0&&iy<IH&&ix>=0&&ix<IW)v+=in[(((b*C+c)*ID+iz)*IH+iy)*IW+ix]*(dz?az:1-az)*(dy?ay:1-ay)*(dx?ax:1-ax);}ref[(((b*C+c)*OD+z)*OH+y)*OW+w]=v;}}for(int run=0;run<4;run++)FUNCTION(in,g,out);for(size_t i=0;i<no;i++)if(fabsf(out[i]-ref[i])>4e-5f*fmaxf(1,fabsf(ref[i]))){fprintf(stderr,"FAIL i=%zu got=%g want=%g\n",i,out[i],ref[i]);return 1;}puts("RESULT correctness=PASS");return 0;}
