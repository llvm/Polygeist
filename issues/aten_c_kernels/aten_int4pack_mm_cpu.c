#define M 32
#define K 64
#define N 48
void aten_int4pack_mm_cpu(float a[M][K],unsigned char packed[N][K/2],float scale[N],float zero[N],float out[M][N]){for(int m=0;m<M;++m)for(int n=0;n<N;++n){float s=0;for(int k=0;k<K;++k){int q=(packed[n][k/2]>>(4*(k&1)))&15;s+=a[m][k]*((float)q-zero[n])*scale[n];}out[m][n]=s;}}
