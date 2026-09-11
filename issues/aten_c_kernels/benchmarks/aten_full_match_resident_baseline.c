/* Device-resident CUDA/cuBLAS/cuDNN baselines for the exhaustive ATen
 * FULL/FULL batch.  This is ordinary C (no custom CUDA kernels), so it can be
 * cross-compiled with aarch64-linux-gnu-gcc against the Jetson SDK stubs. */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Shape names are activated only after vendor headers; short macros such as
 * B, C, and N otherwise rewrite parameter names inside those headers. */
#ifdef ATEN_B
#define B ATEN_B
#endif
#ifdef ATEN_C
#define C ATEN_C
#endif
#ifdef ATEN_D
#define D ATEN_D
#endif
#ifdef ATEN_H
#define H ATEN_H
#endif
#ifdef ATEN_W
#define W ATEN_W
#endif
#ifdef ATEN_IC
#define IC ATEN_IC
#endif
#ifdef ATEN_OC
#define OC ATEN_OC
#endif
#ifdef ATEN_O
#define O ATEN_O
#endif
#ifdef ATEN_K
#define K ATEN_K
#endif
#ifdef ATEN_M
#define M ATEN_M
#endif
#ifdef ATEN_N
#define N ATEN_N
#endif
#ifdef ATEN_R
#define R ATEN_R
#endif
#ifdef ATEN_L
#define L ATEN_L
#endif
#ifdef ATEN_S
#define S ATEN_S
#endif

#ifndef BENCH_ITERS
#define BENCH_ITERS 20
#endif
#define STR1(x) #x
#define STR(x) STR1(x)
#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));return 2;} } while(0)
#define BLAS_OK(x) do { cublasStatus_t e=(x); if(e!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS: %d\n",(int)e);return 2;} } while(0)
#define DNN_OK(x) do { cudnnStatus_t e=(x); if(e!=CUDNN_STATUS_SUCCESS){fprintf(stderr,"cuDNN: %s\n",cudnnGetErrorString(e));return 2;} } while(0)

static void fill_f32(float *p,size_t n,int salt){for(size_t i=0;i<n;++i)p[i]=(float)((int)((i*17+(size_t)salt*13+5)%101)-50)/257.0f;}
static void fill_f64(double *p,size_t n,int salt){for(size_t i=0;i<n;++i)p[i]=(double)((int)((i*17+(size_t)salt*13+5)%101)-50)/257.0;}
static double err_f32(const float*a,const float*b,size_t n){double e=0;for(size_t i=0;i<n;++i)e=fmax(e,fabs((double)a[i]-b[i]));return e;}
static double err_f64(const double*a,const double*b,size_t n){double e=0;for(size_t i=0;i<n;++i)e=fmax(e,fabs(a[i]-b[i]));return e;}
static float elapsed(cudaEvent_t a,cudaEvent_t b){float ms=0;cudaEventElapsedTime(&ms,a,b);return 1000.0f*ms/BENCH_ITERS;}
#define TIME_LAUNCH(stmt) do { \
  cudaEvent_t _a,_b; CUDA_OK(cudaEventCreate(&_a)); CUDA_OK(cudaEventCreate(&_b)); \
  stmt; CUDA_OK(cudaDeviceSynchronize()); CUDA_OK(cudaEventRecord(_a,0)); \
  for(int _i=0;_i<BENCH_ITERS;++_i){stmt;} CUDA_OK(cudaEventRecord(_b,0)); \
  CUDA_OK(cudaEventSynchronize(_b)); us=elapsed(_a,_b); \
} while(0)
#define PRINT_RESULT(err, tol) do { \
  printf("kernel=%s correctness=%s max_abs=%.17g iterations=%d resident_cuda_us=%.6f\n", \
         STR(FUNCTION),(err)<=(tol)?"PASS":"FAIL",(double)(err),BENCH_ITERS,us); \
  return (err)<=(tol)?0:1; \
} while(0)

#if defined(BENCH_KIND_COPY1) || defined(BENCH_KIND_COPY2)
extern void REFERENCE(float*,float*);
int main(void){
#ifdef BENCH_KIND_COPY1
  size_t n=N;
#else
  size_t n=(size_t)B*N;
#endif
  size_t z=n*sizeof(float);float*h=malloc(z),*o=malloc(z),*ref=malloc(z),*dx,*dy;fill_f32(h,n,1);REFERENCE(h,ref);
  CUDA_OK(cudaMalloc((void**)&dx,z));CUDA_OK(cudaMalloc((void**)&dy,z));CUDA_OK(cudaMemcpy(dx,h,z,cudaMemcpyHostToDevice));
  float us;TIME_LAUNCH(cudaMemcpyAsync(dy,dx,z,cudaMemcpyDeviceToDevice,0));CUDA_OK(cudaMemcpy(o,dy,z,cudaMemcpyDeviceToHost));double e=err_f32(o,ref,n);PRINT_RESULT(e,0.0);
}
#elif defined(BENCH_KIND_TWO_COPY)
extern void REFERENCE(float*,float*,float*,float*);
int main(void){size_t z=(size_t)N*4;float*a=malloc(z),*b=malloc(z),*x=malloc(z),*y=malloc(z),*rx=malloc(z),*ry=malloc(z),*da,*db,*dx,*dy;fill_f32(a,N,1);fill_f32(b,N,2);REFERENCE(a,b,rx,ry);CUDA_OK(cudaMalloc((void**)&da,z));CUDA_OK(cudaMalloc((void**)&db,z));CUDA_OK(cudaMalloc((void**)&dx,z));CUDA_OK(cudaMalloc((void**)&dy,z));CUDA_OK(cudaMemcpy(da,a,z,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(db,b,z,cudaMemcpyHostToDevice));float us;TIME_LAUNCH((cudaMemcpyAsync(dx,da,z,cudaMemcpyDeviceToDevice,0),cudaMemcpyAsync(dy,db,z,cudaMemcpyDeviceToDevice,0)));CUDA_OK(cudaMemcpy(x,dx,z,cudaMemcpyDeviceToHost));CUDA_OK(cudaMemcpy(y,dy,z,cudaMemcpyDeviceToHost));double e=fmax(err_f32(x,rx,N),err_f32(y,ry,N));PRINT_RESULT(e,0.0);}
#elif defined(BENCH_KIND_AS_COMPLEX)
extern void REFERENCE(float*,float*,float*);
int main(void){size_t zi=(size_t)N*2*4,zo=(size_t)N*4;float*x=malloc(zi),*re=malloc(zo),*im=malloc(zo),*rr=malloc(zo),*ri=malloc(zo),*dx,*dr,*di;fill_f32(x,(size_t)N*2,1);REFERENCE(x,rr,ri);CUDA_OK(cudaMalloc((void**)&dx,zi));CUDA_OK(cudaMalloc((void**)&dr,zo));CUDA_OK(cudaMalloc((void**)&di,zo));CUDA_OK(cudaMemcpy(dx,x,zi,cudaMemcpyHostToDevice));float us;TIME_LAUNCH((cudaMemcpy2DAsync(dr,4,dx,8,4,N,cudaMemcpyDeviceToDevice,0),cudaMemcpy2DAsync(di,4,dx+1,8,4,N,cudaMemcpyDeviceToDevice,0)));CUDA_OK(cudaMemcpy(re,dr,zo,cudaMemcpyDeviceToHost));CUDA_OK(cudaMemcpy(im,di,zo,cudaMemcpyDeviceToHost));double e=fmax(err_f32(re,rr,N),err_f32(im,ri,N));PRINT_RESULT(e,0.0);}
#elif defined(BENCH_KIND_CAT)
extern void REFERENCE(float*,float*,float*);
int main(void){size_t na=(size_t)R*K,nb=(size_t)M*K,no=na+nb;float*a=malloc(na*4),*b=malloc(nb*4),*o=malloc(no*4),*r=malloc(no*4),*da,*db,*d;fill_f32(a,na,1);fill_f32(b,nb,2);REFERENCE(a,b,r);CUDA_OK(cudaMalloc((void**)&da,na*4));CUDA_OK(cudaMalloc((void**)&db,nb*4));CUDA_OK(cudaMalloc((void**)&d,no*4));CUDA_OK(cudaMemcpy(da,a,na*4,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(db,b,nb*4,cudaMemcpyHostToDevice));float us;TIME_LAUNCH((cudaMemcpyAsync(d,da,na*4,cudaMemcpyDeviceToDevice,0),cudaMemcpyAsync(d+na,db,nb*4,cudaMemcpyDeviceToDevice,0)));CUDA_OK(cudaMemcpy(o,d,no*4,cudaMemcpyDeviceToHost));double e=err_f32(o,r,no);PRINT_RESULT(e,0.0);}
#elif defined(BENCH_KIND_NARROW)
extern void REFERENCE(float*,float*);
int main(void){size_t ni=(size_t)R*C,no=(size_t)R*L;float*x=malloc(ni*4),*o=malloc(no*4),*r=malloc(no*4),*dx,*dy;fill_f32(x,ni,1);REFERENCE(x,r);CUDA_OK(cudaMalloc((void**)&dx,ni*4));CUDA_OK(cudaMalloc((void**)&dy,no*4));CUDA_OK(cudaMemcpy(dx,x,ni*4,cudaMemcpyHostToDevice));float us;TIME_LAUNCH(cudaMemcpy2DAsync(dy,L*4,dx+S,C*4,L*4,R,cudaMemcpyDeviceToDevice,0));CUDA_OK(cudaMemcpy(o,dy,no*4,cudaMemcpyDeviceToHost));double e=err_f32(o,r,no);PRINT_RESULT(e,0.0);}
#elif defined(BENCH_KIND_ZERO)
extern void REFERENCE(float*);
int main(void){size_t z=(size_t)N*4;float*o=malloc(z),*r=malloc(z),*d;fill_f32(r,N,1);REFERENCE(r);CUDA_OK(cudaMalloc((void**)&d,z));float us;TIME_LAUNCH(cudaMemsetAsync(d,0,z,0));CUDA_OK(cudaMemcpy(o,d,z,cudaMemcpyDeviceToHost));double e=err_f32(o,r,N);PRINT_RESULT(e,0.0);}
#elif defined(BENCH_KIND_DOT)
extern void REFERENCE(float*,float*,float*);
int main(void){size_t z=(size_t)K*4;float*a=malloc(z),*b=malloc(z),r=0,o=0,*da,*db,*d;for(size_t i=0;i<K;++i){a[i]=(i&1)?-1.0f:1.0f;b[i]=a[i];}REFERENCE(a,b,&r);cublasHandle_t h;BLAS_OK(cublasCreate(&h));BLAS_OK(cublasSetPointerMode(h,CUBLAS_POINTER_MODE_DEVICE));CUDA_OK(cudaMalloc((void**)&da,z));CUDA_OK(cudaMalloc((void**)&db,z));CUDA_OK(cudaMalloc((void**)&d,4));CUDA_OK(cudaMemcpy(da,a,z,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(db,b,z,cudaMemcpyHostToDevice));float us;TIME_LAUNCH(cublasSdot(h,K,da,1,db,1,d));CUDA_OK(cudaMemcpy(&o,d,4,cudaMemcpyDeviceToHost));double e=fabs((double)o-r);PRINT_RESULT(e,0.0);}
#elif defined(BENCH_KIND_GEMV)
extern void REFERENCE(float*,float*,float*);
int main(void){size_t na=(size_t)M*K;
#ifdef GEMV_TRANS
size_t nx=M,ny=K;
#else
size_t nx=K,ny=M;
#endif
float*a=malloc(na*4),*x=malloc(nx*4),*o=malloc(ny*4),*r=malloc(ny*4),*da,*dx,*dy;fill_f32(a,na,1);fill_f32(x,nx,2);REFERENCE(a,x,r);cublasHandle_t h;BLAS_OK(cublasCreate(&h));CUDA_OK(cudaMalloc((void**)&da,na*4));CUDA_OK(cudaMalloc((void**)&dx,nx*4));CUDA_OK(cudaMalloc((void**)&dy,ny*4));CUDA_OK(cudaMemcpy(da,a,na*4,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(dx,x,nx*4,cudaMemcpyHostToDevice));float one=1,zero=0;float us;
#ifdef GEMV_TRANS
TIME_LAUNCH(cublasSgemv(h,CUBLAS_OP_N,K,M,&one,da,K,dx,1,&zero,dy,1));
#else
TIME_LAUNCH(cublasSgemv(h,CUBLAS_OP_T,K,M,&one,da,K,dx,1,&zero,dy,1));
#endif
CUDA_OK(cudaMemcpy(o,dy,ny*4,cudaMemcpyDeviceToHost));double e=err_f32(o,r,ny);PRINT_RESULT(e,1e-3);}
#elif defined(BENCH_KIND_BATCH_GEMM)
extern void REFERENCE(float*,float*,float*);
int main(void){size_t na=(size_t)B*M*K,nb=(size_t)K*N,no=(size_t)B*M*N;float*a=malloc(na*4),*b=malloc(nb*4),*o=malloc(no*4),*r=malloc(no*4),*da,*db,*dc;fill_f32(a,na,1);fill_f32(b,nb,2);REFERENCE(a,b,r);cublasHandle_t h;BLAS_OK(cublasCreate(&h));CUDA_OK(cudaMalloc((void**)&da,na*4));CUDA_OK(cudaMalloc((void**)&db,nb*4));CUDA_OK(cudaMalloc((void**)&dc,no*4));CUDA_OK(cudaMemcpy(da,a,na*4,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(db,b,nb*4,cudaMemcpyHostToDevice));float one=1,zero=0;float us;TIME_LAUNCH(cublasSgemmStridedBatched(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&one,db,N,0,da,K,(long long)M*K,&zero,dc,N,(long long)M*N,B));CUDA_OK(cudaMemcpy(o,dc,no*4,cudaMemcpyDeviceToHost));double e=err_f32(o,r,no);PRINT_RESULT(e,1e-3);}
#elif defined(BENCH_KIND_LINEAR_COMB)
extern void REFERENCE(float*,float*,float*);
int main(void){size_t ni=(size_t)4*N,no=N;float*x=malloc(ni*4),c[4],*o=malloc(no*4),*r=malloc(no*4),*dx,*dc,*dy;fill_f32(x,ni,1);fill_f32(c,4,2);REFERENCE(x,c,r);cublasHandle_t h;BLAS_OK(cublasCreate(&h));CUDA_OK(cudaMalloc((void**)&dx,ni*4));CUDA_OK(cudaMalloc((void**)&dc,16));CUDA_OK(cudaMalloc((void**)&dy,no*4));CUDA_OK(cudaMemcpy(dx,x,ni*4,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(dc,c,16,cudaMemcpyHostToDevice));float one=1,zero=0;float us;TIME_LAUNCH(cublasSgemv(h,CUBLAS_OP_N,N,4,&one,dx,N,dc,1,&zero,dy,1));CUDA_OK(cudaMemcpy(o,dy,no*4,cudaMemcpyDeviceToHost));double e=err_f32(o,r,no);PRINT_RESULT(e,2e-4);}
#elif defined(BENCH_KIND_OUTER)
extern void REFERENCE(double*,double*,double*);
int main(void){size_t no=(size_t)M*N;double*x=malloc(M*8),*y=malloc(N*8),*o=malloc(no*8),*r=malloc(no*8),*dx,*dy,*dc;fill_f64(x,M,1);fill_f64(y,N,2);REFERENCE(x,y,r);cublasHandle_t h;BLAS_OK(cublasCreate(&h));CUDA_OK(cudaMalloc((void**)&dx,M*8));CUDA_OK(cudaMalloc((void**)&dy,N*8));CUDA_OK(cudaMalloc((void**)&dc,no*8));CUDA_OK(cudaMemcpy(dx,x,M*8,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(dy,y,N*8,cudaMemcpyHostToDevice));double one=1,zero=0;float us;TIME_LAUNCH(cublasDgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,1,&one,dy,N,dx,1,&zero,dc,N));CUDA_OK(cudaMemcpy(o,dc,no*8,cudaMemcpyDeviceToHost));double e=err_f64(o,r,no);PRINT_RESULT(e,1e-12);}
#elif defined(BENCH_KIND_CONV3D_BIAS) || defined(BENCH_KIND_CONV3D) || \
      defined(BENCH_KIND_CONV3D_TRANSPOSE_BACKWARD)
#if defined(BENCH_KIND_CONV3D_BIAS)
#define CIN IC
#define COUT OC
#define ID D
#define IH H
#define IW W
extern void REFERENCE(float*,float*,float*,float*);
#elif defined(BENCH_KIND_CONV3D)
#define CIN C
#define COUT O
#define ID D
#define IH H
#define IW W
extern void REFERENCE(float*,float*,float*);
#else
#define CIN O
#define COUT C
#define ID (D+2)
#define IH (H+2)
#define IW (W+2)
extern void REFERENCE(float*,float*,float*);
#endif
typedef struct {cudnnHandle_t h;cudnnTensorDescriptor_t xd,yd,bd;cudnnFilterDescriptor_t fd;cudnnConvolutionDescriptor_t cd;cudnnConvolutionFwdAlgo_t algo;float*dx,*dw,*db,*dy;void*ws;size_t wsz;} Conv;
static void launch_conv(Conv*c){float one=1,zero=0;cudnnConvolutionForward(c->h,&one,c->xd,c->dx,c->fd,c->dw,c->cd,c->algo,c->ws,c->wsz,&zero,c->yd,c->dy);
#ifdef BENCH_KIND_CONV3D_BIAS
cudnnAddTensor(c->h,&one,c->bd,c->db,&one,c->yd,c->dy);
#endif
}
int main(void){int od=ID-K+1,oh=IH-K+1,ow=IW-K+1;size_t nx=(size_t)CIN*ID*IH*IW,nw=(size_t)COUT*CIN*K*K*K,ny=(size_t)COUT*od*oh*ow;float*x=malloc(nx*4),*w=malloc(nw*4),*o=malloc(ny*4),*r=malloc(ny*4),*bias=NULL;fill_f32(x,nx,1);fill_f32(w,nw,2);
#ifdef BENCH_KIND_CONV3D_BIAS
bias=malloc(COUT*4);fill_f32(bias,COUT,3);REFERENCE(x,w,bias,r);
#else
REFERENCE(x,w,r);
#endif
Conv c={0};DNN_OK(cudnnCreate(&c.h));DNN_OK(cudnnCreateTensorDescriptor(&c.xd));DNN_OK(cudnnCreateTensorDescriptor(&c.yd));DNN_OK(cudnnCreateFilterDescriptor(&c.fd));DNN_OK(cudnnCreateConvolutionDescriptor(&c.cd));int xd[5]={1,CIN,ID,IH,IW},xs[5]={CIN*ID*IH*IW,ID*IH*IW,IH*IW,IW,1};int yd[5]={1,COUT,od,oh,ow},ys[5]={COUT*od*oh*ow,od*oh*ow,oh*ow,ow,1};int fd[5]={COUT,CIN,K,K,K},pad[3]={0,0,0},stride[3]={1,1,1},dilation[3]={1,1,1};DNN_OK(cudnnSetTensorNdDescriptor(c.xd,CUDNN_DATA_FLOAT,5,xd,xs));DNN_OK(cudnnSetTensorNdDescriptor(c.yd,CUDNN_DATA_FLOAT,5,yd,ys));DNN_OK(cudnnSetFilterNdDescriptor(c.fd,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,5,fd));DNN_OK(cudnnSetConvolutionNdDescriptor(c.cd,3,pad,stride,dilation,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT));int got=0;cudnnConvolutionFwdAlgoPerf_t perf;DNN_OK(cudnnGetConvolutionForwardAlgorithm_v7(c.h,c.xd,c.fd,c.cd,c.yd,1,&got,&perf));if(!got)return 2;c.algo=perf.algo;DNN_OK(cudnnGetConvolutionForwardWorkspaceSize(c.h,c.xd,c.fd,c.cd,c.yd,c.algo,&c.wsz));CUDA_OK(cudaMalloc((void**)&c.dx,nx*4));CUDA_OK(cudaMalloc((void**)&c.dw,nw*4));CUDA_OK(cudaMalloc((void**)&c.dy,ny*4));if(c.wsz)CUDA_OK(cudaMalloc(&c.ws,c.wsz));CUDA_OK(cudaMemcpy(c.dx,x,nx*4,cudaMemcpyHostToDevice));CUDA_OK(cudaMemcpy(c.dw,w,nw*4,cudaMemcpyHostToDevice));
#ifdef BENCH_KIND_CONV3D_BIAS
DNN_OK(cudnnCreateTensorDescriptor(&c.bd));int bd[5]={1,COUT,1,1,1},bs[5]={COUT,1,1,1,1};DNN_OK(cudnnSetTensorNdDescriptor(c.bd,CUDNN_DATA_FLOAT,5,bd,bs));CUDA_OK(cudaMalloc((void**)&c.db,COUT*4));CUDA_OK(cudaMemcpy(c.db,bias,COUT*4,cudaMemcpyHostToDevice));
#endif
float us;TIME_LAUNCH(launch_conv(&c));CUDA_OK(cudaMemcpy(o,c.dy,ny*4,cudaMemcpyDeviceToHost));double e=err_f32(o,r,ny);PRINT_RESULT(e,5e-4);}
#else
#error Unsupported resident benchmark kind
#endif
