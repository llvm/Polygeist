#define B 4
#define N 256
void aten_fast_cat_dim0_cpu(float x[B][N],float out[B*N]){for(int b=0;b<B;++b)for(int i=0;i<N;++i)out[b*N+i]=x[b][i];}
