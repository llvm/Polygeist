#ifndef N
#define N 1024
#endif
void aten_blas_sum_cpu(float x[N],float out[1]){float v=0;for(int i=0;i<N;++i)v+=x[i];out[0]=v;}
