#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <unistd.h>
#define POLYGEIST_V2_MATRIX_CUDA
#include "polygeist_v2_matrix_layout.inc"
#include <polygeist_v2_memref_copy.inc>
#define CU(x) do {cudaError_t s=(x);if(s!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(s));return 20;}}while(0)
#define BL(x) do {if((x)!=CUBLAS_STATUS_SUCCESS)return 21;}while(0)
static void pause_stream(void*){usleep(200000);}
int main(){
 const int n=96;float h[n],out[n],want[n];float *a,*b;
 for(int i=0;i<n;i++){h[i]=i+0.25f;out[i]=want[i]=-7.0f;}
 CU(cudaMalloc((void**)&a,sizeof(h)));CU(cudaMalloc((void**)&b,sizeof(h)));
 CU(cudaMemcpy(a,h,sizeof(h),cudaMemcpyHostToDevice));
 CU(cudaMemcpy(b,out,sizeof(out),cudaMemcpyHostToDevice));
 int64_t sizes[2]={3,4},sa[2]={9,2},sb[2]={8,1};
 polygeist_v2_memref_copy(2,4,sizes,sa,sb,a+2,b+3);
 CU(cudaMemcpy(out,b,sizeof(out),cudaMemcpyDeviceToHost));
 for(int i=0;i<3;i++)for(int j=0;j<4;j++)want[3+i*8+j]=h[2+i*9+j*2];
 if(memcmp(out,want,sizeof(out)))return 1;
 polygeist_v2_memref_copy(2,4,sizes,sa,sb,h+2,b+3);
 polygeist_v2_memref_copy(2,4,sizes,sb,sb,b+3,out+3);
 if(memcmp(out,want,sizeof(out)))return 2;
 // A real queued cuBLAS producer uses a nonblocking stream.
 cublasHandle_t handle;cudaStream_t stream;BL(cublasCreate(&handle));
 CU(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));BL(cublasSetStream(handle,stream));
 float scale=1.0f;BL(cublasSscal(handle,n,&scale,a,1));CU(cudaStreamSynchronize(stream));
 CU(cudaLaunchHostFunc(stream,pause_stream,NULL));scale=2.0f;
 BL(cublasSscal(handle,n,&scale,a,1));
 cudaEvent_t ready;CU(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming));CU(cudaEventRecord(ready,stream));
 if(cudaEventQuery(ready)!=cudaErrorNotReady)return 6;
 int64_t count=n,one=1;
 polygeist_v2_memref_copy(1,4,&count,&one,&one,a,b);
 CU(cudaMemcpy(out,b,sizeof(out),cudaMemcpyDeviceToHost));
 for(int i=0;i<n;i++)if(out[i]!=2*h[i])return 3;
 // Mapped/pinned host buffers need synchronization even for host memcpy.
 for(int registered=0;registered<2;registered++) {
  float *host=NULL,*gpu=NULL;
  if(registered){host=(float*)malloc(sizeof(h));CU(cudaHostRegister(host,sizeof(h),cudaHostRegisterMapped));}
  else CU(cudaHostAlloc((void**)&host,sizeof(h),cudaHostAllocMapped));
  CU(cudaHostGetDevicePointer((void**)&gpu,host,0));
  memcpy(host,h,sizeof(h));
  CU(cudaLaunchHostFunc(stream,pause_stream,NULL));scale=2.0f;
  BL(cublasSscal(handle,n,&scale,gpu,1));CU(cudaEventRecord(ready,stream));
  if(cudaEventQuery(ready)!=cudaErrorNotReady)return 8;
  count=n;polygeist_v2_memref_copy(1,4,&count,&one,&one,host,out);
  for(int i=0;i<n;i++)if(out[i]!=2*h[i])return 9+registered;
  // Destination may also have pending GPU writes before a host-side copy.
  CU(cudaLaunchHostFunc(stream,pause_stream,NULL));scale=3.0f;
  BL(cublasSscal(handle,n,&scale,gpu,1));CU(cudaEventRecord(ready,stream));
  if(cudaEventQuery(ready)!=cudaErrorNotReady)return 11;
  polygeist_v2_memref_copy(1,4,&count,&one,&one,h,host);
  CU(cudaStreamSynchronize(stream));
  for(int i=0;i<n;i++)if(host[i]!=h[i])return 12+registered;
  if(registered){CU(cudaHostUnregister(host));free(host);}else CU(cudaFreeHost(host));
 }
 // Snapshot an overlapping device view before writing it back.
 count=32;polygeist_v2_memref_copy(1,4,&count,&one,&one,b,b+3);
 CU(cudaMemcpy(out,b,sizeof(out),cudaMemcpyDeviceToHost));
 for(int i=0;i<n;i++)if(out[i]!=(i>=3&&i<35?2*h[i-3]:2*h[i]))return 4;
 int64_t reverse=-1;count=8;polygeist_v2_memref_copy(1,4,&count,&reverse,&one,a+7,out);
 for(int i=0;i<8;i++)if(out[i]!=2*h[7-i])return 7;
 CU(cudaEventDestroy(ready));count=0;polygeist_v2_memref_copy(1,4,&count,&one,&one,NULL,NULL);
 CU(cudaMemcpy(out,a,sizeof(out),cudaMemcpyDeviceToHost));
 for(int i=0;i<n;i++)if(out[i]!=2*h[i])return 5;
 BL(cublasDestroy(handle));CU(cudaStreamDestroy(stream));CU(cudaFree(a));CU(cudaFree(b));
 puts("PASS H2D/D2H/D2D copies, strided holes, overlap, source preservation and queued cuBLAS producers for device/mapped/registered-host buffers");return 0;
}
