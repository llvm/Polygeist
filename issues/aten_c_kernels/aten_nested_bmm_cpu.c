#define B 8
#define M 16
#define K 24
#define N 20
void aten_nested_bmm_cpu(float a[B][M][K],float b[B][K][N],float out[B][M][N]){for(int q=0;q<B;++q)for(int i=0;i<M;++i)for(int j=0;j<N;++j){float v=0;for(int k=0;k<K;++k)v+=a[q][i][k]*b[q][k][j];out[q][i][j]=v;}}
