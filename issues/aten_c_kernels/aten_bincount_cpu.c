#define N 1024
#define B 64
void aten_bincount_cpu(int x[N],float w[N],float out[B]){for(int b=0;b<B;++b)out[b]=0;for(int i=0;i<N;++i)out[x[i]]+=w[i];}
