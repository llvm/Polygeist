// Exercise ordered comparisons, first-index ties and pointer residency.
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef int (*reduce_t)(int32_t,int32_t,int32_t,const float*,void*,cudaStream_t);
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
    for(int op=0;op<4;++op) {
      for(int r=0;r<rows;++r) {
        expected[r]=0;float best=x[r*cols];
        for(int c=1;c<cols;++c)if((op&1)?x[r*cols+c]<best:x[r*cols+c]>best) {
          best=x[r*cols+c];expected[r]=c;
        }
        if(op>=2)memcpy(&expected[r],&best,sizeof(best));
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
  typedef int (*prefix_t)(int32_t,const int32_t*,int32_t*,cudaStream_t);
  prefix_t prefix=(prefix_t)dlsym(lib,"polygeist_cub_exclusive_sum1d_i32_cuda");
  if(!prefix)return 2;
  const int lengths[]={0,1,7,64};
  int prefix_cases=0;
  for(int k=0;k<4;++k) {
    int n=lengths[k];int32_t x[64],expected[65],got[65],*dx,*dy;
    expected[0]=0;
    for(int i=0;i<n;++i){x[i]=i%5-2;expected[i+1]=expected[i]+x[i];}
    CUDA(cudaMalloc((void**)&dx,67*sizeof(int32_t)));
    CUDA(cudaMalloc((void**)&dy,68*sizeof(int32_t)));
    if(n)CUDA(cudaMemcpy(dx+3,x,n*sizeof(int32_t),cudaMemcpyHostToDevice));
    for(int mask=0;mask<4;++mask) {
      int rc=prefix(n,(mask&1)?dx+3:x,(mask&2)?dy+3:got,0);
      if(rc)return 1;
      if(mask&2)CUDA(cudaMemcpy(got,dy+3,(n+1)*sizeof(int32_t),cudaMemcpyDeviceToHost));
      for(int i=0;i<=n;++i)if(got[i]!=expected[i]) {
        fprintf(stderr,"prefix mismatch n=%d mask=%d index=%d\n",n,mask,i);return 1;
      }
      ++prefix_cases;
    }
    CUDA(cudaFree(dx));CUDA(cudaFree(dy));
  }
  printf("PASS CUB prefix residency: %d cases\n",prefix_cases);
  printf("PASS CUB index/value reduction residency: %d cases\n",cases);
  return 0;
}
