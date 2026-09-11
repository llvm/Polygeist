#define M 64
#define K 96
void aten_blas_gemv_generic_cpu(float a[M][K],float x[K],float out[M]){for(int i=0;i<M;++i){float v=0;for(int k=0;k<K;++k)v+=a[i][k]*x[k];out[i]=v;}}
