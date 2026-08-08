#define B 4
#define M 16
#define N 20
#define K 24
void aten_cpu_blas_gemm_strided_batched_cpu(float a[B][M][K],float b[B][K][N],float c[B][M][N]){for(int q=0;q<B;++q)for(int i=0;i<M;++i)for(int j=0;j<N;++j){float v=0;for(int k=0;k<K;++k)v+=a[q][i][k]*b[q][k][j];c[q][i][j]=v;}}
