#define M 32
#define N 48
#define K 40
void aten_gemm_transab_cpu(float a[M][K],float b[K][N],float c[M][N]){for(int i=0;i<M;++i)for(int j=0;j<N;++j)for(int k=0;k<K;++k)c[i][j]+=a[k][i]*b[j][k];}
