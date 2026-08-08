#define M 32
#define K 64
#define N 48
void aten_int8pack_mm_cpu(float a[M][K],signed char weight[N][K],float scale[N],float out[M][N]){for(int m=0;m<M;++m)for(int n=0;n<N;++n){out[m][n]=0;for(int k=0;k<K;++k)out[m][n]+=a[m][k]*(float)weight[n][k]*scale[n];}}
