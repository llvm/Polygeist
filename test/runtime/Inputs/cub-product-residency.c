// Complete-output regression for CUB scan argument residency and dead outputs.
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef int (*scan_t)(int32_t, int32_t, const float *, float *, float *, cudaStream_t);
#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e)); return 2; } } while(0)
int main(int argc,char **argv) {
  if(argc!=2)return 2;
  void *lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);
  scan_t scan=lib?(scan_t)dlsym(lib,"polygeist_cub_segmented_inclusive_product2d_f32_cuda"):NULL;
  if(!scan){fprintf(stderr,"CUB load failed: %s\n",dlerror());return 2;}
  int cases=0;
  for(int columns=11;columns<=37;columns+=26) {
    const int rows=5, count=rows*columns;
    float *input=malloc(count*sizeof(float)), *expected=malloc(count*sizeof(float));
    float *actual=malloc(count*sizeof(float)), final[5];
    float *di,*dout,*df;
    CUDA(cudaMalloc((void**)&di,(count+3)*sizeof(float)));
    CUDA(cudaMalloc((void**)&dout,(count+3)*sizeof(float)));
    CUDA(cudaMalloc((void**)&df,rows*sizeof(float)));
    for(int r=0;r<rows;++r) {
      float value=1;
      for(int c=0;c<columns;++c) {
        int i=r*columns+c; input[i]=0.99f+0.002f*(i%11);
        value*=input[i]; expected[i]=value;
      }
    }
    for(int mask=0;mask<16;++mask) {
      // All host/device combinations, including null auxiliary output and
      // nonzero offsets in device allocations. Repeat to exercise caching.
      for(int repeat=0;repeat<3;++repeat) {
        CUDA(cudaMemcpy(di+3,input,count*sizeof(float),cudaMemcpyHostToDevice));
        float *out=(mask&2)?dout+3:actual;
        float *aux=(mask&8)?NULL:((mask&4)?df:final);
        int rc=scan(rows,columns,(mask&1)?di+3:input,aux,out,0);
        if(rc){fprintf(stderr,"scan failed mask=%d status=%d\n",mask,rc);return 1;}
        if(mask&2)CUDA(cudaMemcpy(actual,out,count*sizeof(float),cudaMemcpyDeviceToHost));
        if(aux && (mask&4))CUDA(cudaMemcpy(final,df,rows*sizeof(float),cudaMemcpyDeviceToHost));
        for(int i=0;i<count;++i)if(!isfinite(actual[i])||fabsf(actual[i]-expected[i])>1e-5f){fprintf(stderr,"output mismatch mask=%d index=%d\n",mask,i);return 1;}
        if(aux)for(int r=0;r<rows;++r)if(fabsf(final[r]-expected[(r+1)*columns-1])>1e-5f)return 1;
        ++cases;
      }
    }
    CUDA(cudaFree(di));CUDA(cudaFree(dout));CUDA(cudaFree(df));
    free(actual);free(expected);free(input);
  }
  printf("PASS CUB product residency: %d cases\n",cases);
  return 0;
}
