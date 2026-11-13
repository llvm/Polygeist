// Full runtime composition, including NaNs, infinities and offset pointers.
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern int cudaMalloc(void **, unsigned long);
extern int cudaFree(void *);
extern int cudaMemcpy(void *, const void *, unsigned long, int);
extern int cudaDeviceSynchronize(void);
extern void polygeist_cublas_joint_maxabs_product_f32(int,int,const float*,const float*,float*);
#define CUDA(x) do { if((x)!=0) return 2; } while(0)
static float oracle(const float *x,int n) {
  float m=0;
  for(int i=0;i<n;++i) {float a=x[i]<0?-x[i]:x[i];if(a>m)m=a;}
  return m;
}
int main(void) {
  int cases=0;const int widths[]={1,7,64,73};
  for(int w=0;w<4;++w) {
    int na=widths[w],nb=widths[3-w];
    // Keep host buffers alive for the runtime's pinned-allocation cache.
    float *a=calloc(na,sizeof(float)),*b=calloc(nb,sizeof(float));
    float *out=calloc(1,sizeof(float)),*da,*db,*dout;
    CUDA(cudaMalloc((void**)&da,(na+3)*sizeof(float)));
    CUDA(cudaMalloc((void**)&db,(nb+3)*sizeof(float)));
    CUDA(cudaMalloc((void**)&dout,4*sizeof(float)));
    for(int scenario=0;scenario<8;++scenario) {
      for(int j=0;j<2;++j) {
        float *x=j?b:a;int n=j?nb:na;
        for(int i=0;i<n;++i) {
          float v=(float)((i*7)%13-6)*1.234567f;
          if(scenario==1)v=NAN;
          if(scenario==2 && i==0)v=NAN;
          if(scenario==3 && i%2)v=NAN;
          if(scenario==4)v=i%2?-INFINITY:INFINITY;
          if(scenario==5)v=i%2?-0.0f:0.0f;
          if(scenario==6)v=j?0.0f:INFINITY;
          if(scenario==7)v=j?0.5f:-FLT_MAX;
          x[i]=v;
        }
      }
      float expected=oracle(a,na)*oracle(b,nb);
      CUDA(cudaMemcpy(da+3,a,na*sizeof(float),1));
      CUDA(cudaMemcpy(db+3,b,nb*sizeof(float),1));
      for(int mask=0;mask<8;++mask) {
        *out=-7.0f;
        polygeist_cublas_joint_maxabs_product_f32(na,nb,
            mask&1?da+3:a,mask&2?db+3:b,mask&4?dout+3:out);
        CUDA(cudaDeviceSynchronize());
        if(mask&4)CUDA(cudaMemcpy(out,dout+3,sizeof(float),2));
        if(!(isnan(expected)&&isnan(*out)) && memcmp(out,&expected,sizeof(float))) {
          fprintf(stderr,"joint mismatch width=%d scenario=%d mask=%d got=%g expected=%g\n",na,scenario,mask,*out,expected);return 1;
        }
        ++cases;
      }
    }
    CUDA(cudaFree(da));CUDA(cudaFree(db));CUDA(cudaFree(dout));
  }
  printf("PASS joint maxabs residency: %d cases\n",cases);return 0;
}
