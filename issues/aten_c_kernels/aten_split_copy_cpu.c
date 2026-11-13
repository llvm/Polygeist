#define N 128
#define S 4
void aten_split_copy_cpu(float x[N],float out[S][N/S]){for(int s=0;s<S;++s)for(int i=0;i<N/S;++i)out[s][i]=x[s*(N/S)+i];}
