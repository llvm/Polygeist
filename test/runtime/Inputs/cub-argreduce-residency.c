// Exercise ordered comparisons, first-index ties and pointer residency.
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
typedef int (*reduce_t)(int32_t,int32_t,int32_t,const float*,int32_t*,cudaStream_t);
#define CUDA(x) do { if((x)!=cudaSuccess) return 2; } while(0)
int main(int argc,char **argv) {
  if(argc!=2)return 2;
  void *lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
  reduce_t reduce=lib?(reduce_t)dlsym(lib,"polygeist_cub_segmented_argreduce_f32_cuda"):NULL;
  if(!reduce){fprintf(stderr,"CUB load failed: %s\n",dlerror());return 2;}
  int cases=0;
  const int widths[]={1,7,64,73};
  for(int w=0;w<4;++w) {
    int rows=8,cols=widths[w],count=rows*cols;
    float *x=malloc(count*sizeof(float)),*dx;
    int32_t expected[8],got[8],*dout;
    CUDA(cudaMalloc((void**)&dx,(count+3)*sizeof(float)));
    CUDA(cudaMalloc((void**)&dout,11*sizeof(int32_t)));
    for(int r=0;r<rows;++r)for(int c=0;c<cols;++c) {
      float v=(float)((c*7)%11-5);
      if(r==0)v=2; // equal extrema: first index wins
      if(r==1 && c==0)v=NAN;
      if(r==2 && c>0)v=NAN;
      if(r==3)v=NAN;
      if(r==4)v=c%2?INFINITY:-INFINITY;
      if(r==5)v=c%2?-0.0f:0.0f;
      if(r==6 && c==0)v=100;
      if(r==7 && c==0)v=-100;
      x[r*cols+c]=v;
    }
    CUDA(cudaMemcpy(dx+3,x,count*sizeof(float),cudaMemcpyHostToDevice));
    for(int op=0;op<2;++op) {
      for(int r=0;r<rows;++r) {
        expected[r]=0;float best=x[r*cols];
        for(int c=1;c<cols;++c)if(op?x[r*cols+c]<best:x[r*cols+c]>best) {
          best=x[r*cols+c];expected[r]=c;
        }
      }
      for(int mask=0;mask<4;++mask) {
        int rc=reduce(op,rows,cols,(mask&1)?dx+3:x,(mask&2)?dout+3:got,0);
        if(rc){fprintf(stderr,"reduce status=%d\n",rc);return 1;}
        if(mask&2)CUDA(cudaMemcpy(got,dout+3,sizeof(got),cudaMemcpyDeviceToHost));
        for(int r=0;r<rows;++r)if(got[r]!=expected[r]) {
          fprintf(stderr,"mismatch op=%d width=%d mask=%d row=%d got=%d expected=%d\n",op,cols,mask,r,got[r],expected[r]);return 1;
        }
        ++cases;
      }
    }
    CUDA(cudaFree(dx));CUDA(cudaFree(dout));free(x);
  }
  printf("PASS CUB arg reduction residency: %d cases\n",cases);
  return 0;
}
