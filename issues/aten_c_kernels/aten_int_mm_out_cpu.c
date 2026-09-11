#define M 32
#define N 48
#define K 64
void aten_int_mm_out_cpu(signed char a[M][K],signed char b[K][N],int out[M][N]){for(int i=0;i<M;++i)for(int j=0;j<N;++j){int v=0;for(int k=0;k<K;++k)v+=(int)a[i][k]*(int)b[k][j];out[i][j]=v;}}
