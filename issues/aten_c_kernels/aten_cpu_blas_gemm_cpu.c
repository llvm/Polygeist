#define M 24
#define N 32
#define K 40
void aten_cpu_blas_gemm_cpu(float a[M][K],float b[K][N],float c[M][N]){for(int i=0;i<M;++i)for(int j=0;j<N;++j){float v=0;for(int k=0;k<K;++k)v+=a[i][k]*b[k][j];c[i][j]=v;}}
