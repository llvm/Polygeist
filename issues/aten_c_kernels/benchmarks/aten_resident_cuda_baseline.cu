#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Shape macros are defined only after CUDA/cuBLAS/cuDNN headers. Short names
// such as B and C otherwise rewrite parameter names inside vendor headers.
#ifdef ATEN_B
#define B ATEN_B
#endif
#ifdef ATEN_C
#define C ATEN_C
#endif
#ifdef ATEN_H
#define H ATEN_H
#endif
#ifdef ATEN_W
#define W ATEN_W
#endif
#ifdef ATEN_M
#define M ATEN_M
#endif
#ifdef ATEN_N
#define N ATEN_N
#endif
#ifdef ATEN_K
#define K ATEN_K
#endif
#ifdef ATEN_IC
#define IC ATEN_IC
#endif
#ifdef ATEN_OC
#define OC ATEN_OC
#endif
#ifdef ATEN_KH
#define KH ATEN_KH
#endif
#ifdef ATEN_KW
#define KW ATEN_KW
#endif
#ifdef ATEN_S
#define S ATEN_S
#endif

#ifndef BENCH_ITERS
#define BENCH_ITERS 20
#endif

#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e)); return 2;} } while(0)
#define BLAS_OK(x) do { cublasStatus_t e=(x); if(e!=CUBLAS_STATUS_SUCCESS){std::fprintf(stderr,"cuBLAS: %d\n",(int)e); return 2;} } while(0)
#define DNN_OK(x) do { cudnnStatus_t e=(x); if(e!=CUDNN_STATUS_SUCCESS){std::fprintf(stderr,"cuDNN: %s\n",cudnnGetErrorString(e)); return 2;} } while(0)

static double value(size_t i, int salt) {
  int centered=(int)((i*17+(size_t)salt*13+5)%101)-50;
  return (double)centered/257.0;
}
template<class T> static void fill(std::vector<T>& v,int salt){for(size_t i=0;i<v.size();++i)v[i]=(T)value(i,salt);}
template<class T> static double max_error(const std::vector<T>& a,const std::vector<T>& b){double m=0;for(size_t i=0;i<a.size();++i)m=fmax(m,fabs((double)a[i]-(double)b[i]));return m;}
static float timed(cudaEvent_t begin,cudaEvent_t end,void (*launch)(void*),void *ctx){
  launch(ctx); CUDA_OK(cudaDeviceSynchronize());
  CUDA_OK(cudaEventRecord(begin));
  for(int i=0;i<BENCH_ITERS;++i) launch(ctx);
  CUDA_OK(cudaEventRecord(end)); CUDA_OK(cudaEventSynchronize(end));
  float ms=0; CUDA_OK(cudaEventElapsedTime(&ms,begin,end)); return 1000.0f*ms/BENCH_ITERS;
}

__global__ static void gelu_kernel(const float*x,float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=x[i];float q=.7978845608f*(v+.044715f*v*v*v);y[i]=.5f*v*(1.0f+tanhf(q));}}
__global__ static void rms_partial(const float*x,float*p,int n){__shared__ float s[256];int t=threadIdx.x;int i=2*blockIdx.x*blockDim.x+t;float v=0;if(i<n)v=x[i]*x[i];if(i+blockDim.x<n)v+=x[i+blockDim.x]*x[i+blockDim.x];s[t]=v;__syncthreads();for(int d=128;d;d>>=1){if(t<d)s[t]+=s[t+d];__syncthreads();}if(!t)p[blockIdx.x]=s[0];}
__global__ static void rms_finish(const float*p,float*scale,int n,int count){__shared__ float s[256];int t=threadIdx.x;float v=0;for(int i=t;i<count;i+=256)v+=p[i];s[t]=v;__syncthreads();for(int d=128;d;d>>=1){if(t<d)s[t]+=s[t+d];__syncthreads();}if(!t)*scale=rsqrtf(s[0]/n+1.0e-5f);}
__global__ static void rms_scale(const float*x,const float*w,float*y,const float*scale,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=w[i]*x[i]*(*scale);}

#if defined(BENCH_ATEN_ADD)
extern "C" void aten_add_reference(float*,float*);
struct Ctx{cudnnHandle_t h;cudnnTensorDescriptor_t d;float*a,*o;};
static void launch(void*p){auto*c=(Ctx*)p;float one=1; cudnnAddTensor(c->h,&one,c->d,c->a,&one,c->d,c->o);}
int main(){size_t n=(size_t)B*C*H*W;std::vector<float>a(n),o(n),ref;fill(a,1);fill(o,2);ref=o;aten_add_reference(a.data(),ref.data());Ctx c;DNN_OK(cudnnCreate(&c.h));DNN_OK(cudnnCreateTensorDescriptor(&c.d));DNN_OK(cudnnSetTensor4dDescriptor(c.d,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,B,C,H,W));CUDA_OK(cudaMalloc(&c.a,n*4));CUDA_OK(cudaMalloc(&c.o,n*4));CUDA_OK(cudaMemcpy(c.a,a.data(),n*4,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(c.o,o.data(),n*4,cudaMemcpyHostToDevice));
#elif defined(BENCH_ATEN_BATCH_NORM)
extern "C" void aten_batch_norm_reference(float*,float*,float*,float*,float*,float*);
struct Ctx{cudnnHandle_t h;cudnnTensorDescriptor_t xd,bn;float*dx,*s,*m,*v,*b,*o;};
static void launch(void*p){auto*c=(Ctx*)p;float one=1,zero=0;cudnnBatchNormalizationForwardInference(c->h,CUDNN_BATCHNORM_SPATIAL,&one,&zero,c->xd,c->dx,c->xd,c->o,c->bn,c->s,c->b,c->m,c->v,1e-5);}
int main(){size_t n=(size_t)B*C*H*W;std::vector<float>x(n),s(C),m(C),is(C),b(C),o(n),ref(n),var(C);fill(x,1);fill(s,2);fill(m,3);fill(b,4);for(int i=0;i<C;++i){is[i]=.75f+.005f*(i%41);var[i]=1/(is[i]*is[i])-1e-5f;}aten_batch_norm_reference(x.data(),s.data(),m.data(),is.data(),b.data(),ref.data());Ctx c;DNN_OK(cudnnCreate(&c.h));DNN_OK(cudnnCreateTensorDescriptor(&c.xd));DNN_OK(cudnnCreateTensorDescriptor(&c.bn));DNN_OK(cudnnSetTensor4dDescriptor(c.xd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,B,C,H,W));DNN_OK(cudnnSetTensor4dDescriptor(c.bn,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,1,C,1,1));CUDA_OK(cudaMalloc(&c.dx,n*4));CUDA_OK(cudaMalloc(&c.o,n*4));CUDA_OK(cudaMalloc(&c.s,C*4));CUDA_OK(cudaMalloc(&c.m,C*4));CUDA_OK(cudaMalloc(&c.v,C*4));CUDA_OK(cudaMalloc(&c.b,C*4));CUDA_OK(cudaMemcpy(c.dx,x.data(),n*4,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(c.s,s.data(),C*4,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(c.m,m.data(),C*4,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(c.v,var.data(),C*4,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(c.b,b.data(),C*4,cudaMemcpyHostToDevice));
#elif defined(BENCH_ATEN_CONV2D)
extern "C" void aten_conv2d_reference(float*,float*,float*);
#define OH (H-KH+1)
#define OW (W-KW+1)
struct Ctx{cudnnHandle_t h;cudnnTensorDescriptor_t xd,yd;cudnnFilterDescriptor_t fd;cudnnConvolutionDescriptor_t conv;float*dx,*df,*dy;void*ws;size_t wsz;cudnnConvolutionFwdAlgo_t algo;};
static void launch(void*p){auto*c=(Ctx*)p;float one=1,zero=0;cudnnConvolutionForward(c->h,&one,c->xd,c->dx,c->fd,c->df,c->conv,c->algo,c->ws,c->wsz,&zero,c->yd,c->dy);}
int main(){size_t nx=(size_t)B*IC*H*W,nf=(size_t)OC*IC*KH*KW,ny=(size_t)B*OC*OH*OW;std::vector<float>x(nx),f(nf),o(ny),ref(ny);fill(x,1);fill(f,2);aten_conv2d_reference(x.data(),f.data(),ref.data());Ctx c;DNN_OK(cudnnCreate(&c.h));DNN_OK(cudnnCreateTensorDescriptor(&c.xd));DNN_OK(cudnnCreateTensorDescriptor(&c.yd));DNN_OK(cudnnCreateFilterDescriptor(&c.fd));DNN_OK(cudnnCreateConvolutionDescriptor(&c.conv));DNN_OK(cudnnSetTensor4dDescriptor(c.xd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,B,IC,H,W));DNN_OK(cudnnSetTensor4dDescriptor(c.yd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,B,OC,OH,OW));DNN_OK(cudnnSetFilter4dDescriptor(c.fd,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,OC,IC,KH,KW));DNN_OK(cudnnSetConvolution2dDescriptor(c.conv,0,0,1,1,1,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT));int got=0;cudnnConvolutionFwdAlgoPerf_t perf;DNN_OK(cudnnGetConvolutionForwardAlgorithm_v7(c.h,c.xd,c.fd,c.conv,c.yd,1,&got,&perf));c.algo=perf.algo;DNN_OK(cudnnGetConvolutionForwardWorkspaceSize(c.h,c.xd,c.fd,c.conv,c.yd,c.algo,&c.wsz));CUDA_OK(cudaMalloc(&c.dx,nx*4));CUDA_OK(cudaMalloc(&c.df,nf*4));CUDA_OK(cudaMalloc(&c.dy,ny*4));c.ws=nullptr;if(c.wsz)CUDA_OK(cudaMalloc(&c.ws,c.wsz));CUDA_OK(cudaMemcpy(c.dx,x.data(),nx*4,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(c.df,f.data(),nf*4,cudaMemcpyHostToDevice));
#elif defined(BENCH_ATEN_MAX_POOL2D)
extern "C" void aten_max_pool2d_reference(float*,float*);
#define OH ((H-K)/S+1)
#define OW ((W-K)/S+1)
struct Ctx{cudnnHandle_t h;cudnnTensorDescriptor_t xd,yd;cudnnPoolingDescriptor_t pool;float*dx,*dy;};
static void launch(void*p){auto*c=(Ctx*)p;float one=1,zero=0;cudnnPoolingForward(c->h,c->pool,&one,c->xd,c->dx,&zero,c->yd,c->dy);}
int main(){size_t nx=(size_t)B*C*H*W,ny=(size_t)B*C*OH*OW;std::vector<float>x(nx),o(ny),ref(ny);fill(x,1);aten_max_pool2d_reference(x.data(),ref.data());Ctx c;DNN_OK(cudnnCreate(&c.h));DNN_OK(cudnnCreateTensorDescriptor(&c.xd));DNN_OK(cudnnCreateTensorDescriptor(&c.yd));DNN_OK(cudnnCreatePoolingDescriptor(&c.pool));DNN_OK(cudnnSetTensor4dDescriptor(c.xd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,B,C,H,W));DNN_OK(cudnnSetTensor4dDescriptor(c.yd,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,B,C,OH,OW));DNN_OK(cudnnSetPooling2dDescriptor(c.pool,CUDNN_POOLING_MAX,CUDNN_NOT_PROPAGATE_NAN,K,K,0,0,S,S));CUDA_OK(cudaMalloc(&c.dx,nx*4));CUDA_OK(cudaMalloc(&c.dy,ny*4));CUDA_OK(cudaMemcpy(c.dx,x.data(),nx*4,cudaMemcpyHostToDevice));
#elif defined(BENCH_ATEN_SOFTMAX)
extern "C" void aten_softmax_reference(float*);
struct Ctx{cudnnHandle_t h;cudnnTensorDescriptor_t d;float*x;};static void launch(void*p){auto*c=(Ctx*)p;float one=1,zero=0;cudnnSoftmaxForward(c->h,CUDNN_SOFTMAX_ACCURATE,CUDNN_SOFTMAX_MODE_INSTANCE,&one,c->d,c->x,&zero,c->d,c->x);}
int main(){size_t n=N;std::vector<float>o(n),ref;fill(o,1);ref=o;aten_softmax_reference(ref.data());Ctx c;DNN_OK(cudnnCreate(&c.h));DNN_OK(cudnnCreateTensorDescriptor(&c.d));DNN_OK(cudnnSetTensor4dDescriptor(c.d,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,1,N,1,1));CUDA_OK(cudaMalloc(&c.x,n*4));CUDA_OK(cudaMemcpy(c.x,o.data(),n*4,cudaMemcpyHostToDevice));
#elif defined(BENCH_ATEN_ADDMM) || defined(BENCH_ATEN_MM)
#if defined(BENCH_ATEN_ADDMM)
extern "C" void aten_addmm_reference(double*,double*,double*,double,double);
#else
extern "C" void aten_mm_reference(double*,double*,double*);
#endif
struct Ctx{cublasHandle_t h;double*a,*b,*c;};static void launch(void*p){auto*c=(Ctx*)p;
#if defined(BENCH_ATEN_ADDMM)
double alpha=1.25,beta=.5;
#else
double alpha=1,beta=0;
#endif
cublasDgemm(c->h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,c->b,N,c->a,K,&beta,c->c,N);}
int main(){size_t na=(size_t)M*K,nb=(size_t)K*N,nc=(size_t)M*N;std::vector<double>a(na),b(nb),o(nc),ref;fill(a,1);fill(b,2);fill(o,3);ref=o;
#if defined(BENCH_ATEN_ADDMM)
aten_addmm_reference(a.data(),b.data(),ref.data(),.5,1.25);
#else
aten_mm_reference(a.data(),b.data(),ref.data());
#endif
Ctx c;BLAS_OK(cublasCreate(&c.h));CUDA_OK(cudaMalloc(&c.a,na*8));CUDA_OK(cudaMalloc(&c.b,nb*8));CUDA_OK(cudaMalloc(&c.c,nc*8));CUDA_OK(cudaMemcpy(c.a,a.data(),na*8,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(c.b,b.data(),nb*8,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(c.c,o.data(),nc*8,cudaMemcpyHostToDevice));
#elif defined(BENCH_ATEN_DOT)
extern "C" void aten_dot_reference(double*,double*,double*);struct Ctx{cublasHandle_t h;double*x,*y,*o;};static void launch(void*p){auto*c=(Ctx*)p;cublasDdot(c->h,N,c->x,1,c->y,1,c->o);}int main(){std::vector<double>x(N),y(N),o(1),ref(1);fill(x,1);fill(y,2);aten_dot_reference(x.data(),y.data(),ref.data());Ctx c;BLAS_OK(cublasCreate(&c.h));BLAS_OK(cublasSetPointerMode(c.h,CUBLAS_POINTER_MODE_DEVICE));CUDA_OK(cudaMalloc(&c.x,N*8));CUDA_OK(cudaMalloc(&c.y,N*8));CUDA_OK(cudaMalloc(&c.o,8));CUDA_OK(cudaMemcpy(c.x,x.data(),N*8,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(c.y,y.data(),N*8,cudaMemcpyHostToDevice));
#elif defined(BENCH_ATEN_MV)
extern "C" void aten_mv_reference(double*,double*,double*);struct Ctx{cublasHandle_t h;double*a,*x,*y;};static void launch(void*p){auto*c=(Ctx*)p;double one=1;cublasDgemv(c->h,CUBLAS_OP_T,K,M,&one,c->a,K,c->x,1,&one,c->y,1);}int main(){size_t na=(size_t)M*K;std::vector<double>a(na),x(K),o(M),ref;fill(a,1);fill(x,2);fill(o,3);ref=o;aten_mv_reference(a.data(),x.data(),ref.data());Ctx c;BLAS_OK(cublasCreate(&c.h));CUDA_OK(cudaMalloc(&c.a,na*8));CUDA_OK(cudaMalloc(&c.x,K*8));CUDA_OK(cudaMalloc(&c.y,M*8));CUDA_OK(cudaMemcpy(c.a,a.data(),na*8,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(c.x,x.data(),K*8,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(c.y,o.data(),M*8,cudaMemcpyHostToDevice));
#elif defined(BENCH_ATEN_GELU)
extern "C" void aten_gelu_reference(float*,float*);struct Ctx{float*x,*y;};static void launch(void*p){auto*c=(Ctx*)p;gelu_kernel<<<(N+255)/256,256>>>(c->x,c->y,N);}int main(){std::vector<float>x(N),o(N),ref(N);fill(x,1);aten_gelu_reference(x.data(),ref.data());Ctx c;CUDA_OK(cudaMalloc(&c.x,N*4));CUDA_OK(cudaMalloc(&c.y,N*4));CUDA_OK(cudaMemcpy(c.x,x.data(),N*4,cudaMemcpyHostToDevice));
#elif defined(BENCH_ATEN_RMS_NORM)
extern "C" void aten_rms_norm_reference(float*,float*,float*,float);struct Ctx{float*x,*w,*y,*p,*scale;int blocks;};static void launch(void*p){auto*c=(Ctx*)p;rms_partial<<<c->blocks,256>>>(c->x,c->p,N);rms_finish<<<1,256>>>(c->p,c->scale,N,c->blocks);rms_scale<<<(N+255)/256,256>>>(c->x,c->w,c->y,c->scale,N);}int main(){std::vector<float>x(N),w(N),o(N),ref(N);fill(x,1);fill(w,2);aten_rms_norm_reference(x.data(),w.data(),ref.data(),1e-5f);Ctx c;c.blocks=(N+511)/512;CUDA_OK(cudaMalloc(&c.x,N*4));CUDA_OK(cudaMalloc(&c.w,N*4));CUDA_OK(cudaMalloc(&c.y,N*4));CUDA_OK(cudaMalloc(&c.p,c.blocks*4));CUDA_OK(cudaMalloc(&c.scale,4));CUDA_OK(cudaMemcpy(c.x,x.data(),N*4,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(c.w,w.data(),N*4,cudaMemcpyHostToDevice));
#else
#error Select BENCH_ATEN_*
#endif

  launch(&c); CUDA_OK(cudaDeviceSynchronize());
#if defined(BENCH_ATEN_ADD)
  CUDA_OK(cudaMemcpy(o.data(),c.o,n*4,cudaMemcpyDeviceToHost)); const char*name="aten_add";
#elif defined(BENCH_ATEN_BATCH_NORM)
  CUDA_OK(cudaMemcpy(o.data(),c.o,n*4,cudaMemcpyDeviceToHost)); const char*name="aten_batch_norm";
#elif defined(BENCH_ATEN_CONV2D)
  CUDA_OK(cudaMemcpy(o.data(),c.dy,ny*4,cudaMemcpyDeviceToHost)); const char*name="aten_conv2d";
#elif defined(BENCH_ATEN_MAX_POOL2D)
  CUDA_OK(cudaMemcpy(o.data(),c.dy,ny*4,cudaMemcpyDeviceToHost)); const char*name="aten_max_pool2d";
#elif defined(BENCH_ATEN_SOFTMAX)
  CUDA_OK(cudaMemcpy(o.data(),c.x,n*4,cudaMemcpyDeviceToHost)); const char*name="aten_softmax";
#elif defined(BENCH_ATEN_ADDMM)
  CUDA_OK(cudaMemcpy(o.data(),c.c,nc*8,cudaMemcpyDeviceToHost)); const char*name="aten_addmm";
#elif defined(BENCH_ATEN_MM)
  CUDA_OK(cudaMemcpy(o.data(),c.c,nc*8,cudaMemcpyDeviceToHost)); const char*name="aten_mm";
#elif defined(BENCH_ATEN_DOT)
  CUDA_OK(cudaMemcpy(o.data(),c.o,8,cudaMemcpyDeviceToHost)); const char*name="aten_dot";
#elif defined(BENCH_ATEN_MV)
  CUDA_OK(cudaMemcpy(o.data(),c.y,M*8,cudaMemcpyDeviceToHost)); const char*name="aten_mv";
#elif defined(BENCH_ATEN_GELU)
  CUDA_OK(cudaMemcpy(o.data(),c.y,N*4,cudaMemcpyDeviceToHost)); const char*name="aten_gelu";
#elif defined(BENCH_ATEN_RMS_NORM)
  CUDA_OK(cudaMemcpy(o.data(),c.y,N*4,cudaMemcpyDeviceToHost)); const char*name="aten_rms_norm";
#endif
  double err=max_error(o,ref);
  // Restore mutable operands before timing. Repeated calls then measure only
  // resident GPU execution, with no setup or host/device transfer included.
#if defined(BENCH_ATEN_ADD)
  fill(o,2); CUDA_OK(cudaMemcpy(c.o,o.data(),n*4,cudaMemcpyHostToDevice));
#elif defined(BENCH_ATEN_ADDMM)
  fill(o,3); CUDA_OK(cudaMemcpy(c.c,o.data(),nc*8,cudaMemcpyHostToDevice));
#elif defined(BENCH_ATEN_MV)
  fill(o,3); CUDA_OK(cudaMemcpy(c.y,o.data(),M*8,cudaMemcpyHostToDevice));
#elif defined(BENCH_ATEN_SOFTMAX)
  fill(o,1); CUDA_OK(cudaMemcpy(c.x,o.data(),n*4,cudaMemcpyHostToDevice));
#endif
  cudaEvent_t begin,end; CUDA_OK(cudaEventCreate(&begin)); CUDA_OK(cudaEventCreate(&end));
  float us=timed(begin,end,launch,&c);
  std::printf("kernel=%s correctness=%s max_abs=%.17g iterations=%d resident_cuda_us=%.6f\n",name,err<2e-4?"PASS":"FAIL",err,BENCH_ITERS,us);return err<2e-4?0:1;
}
