#define N 2048
void aten_blas_dot_naive_cpu(float a[N],float b[N],float out[1]){float v=0;for(int i=0;i<N;++i)v+=a[i]*b[i];out[0]=v;}
