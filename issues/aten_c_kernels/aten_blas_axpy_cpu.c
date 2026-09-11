#define N 1024
void aten_blas_axpy_cpu(float a,float x[N],float y[N]){for(int i=0;i<N;++i)y[i]+=a*x[i];}
