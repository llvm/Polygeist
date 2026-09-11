#define B 16
#define M 32
#define K 64
#define N 48
void aten_flatten_nd_linear_cpu(float x[B][M][K],float w[K][N],float out[B][M][N]){for(int b=0;b<B;++b)for(int i=0;i<M;++i)for(int j=0;j<N;++j){float v=0;for(int k=0;k<K;++k)v+=x[b][i][k]*w[k][j];out[b][i][j]=v;}}
