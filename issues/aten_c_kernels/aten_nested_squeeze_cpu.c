#define B 8
#define N 64
void aten_nested_squeeze_cpu(float x[B][1][N],float out[B][N]){for(int b=0;b<B;++b)for(int i=0;i<N;++i)out[b][i]=x[b][0][i];}
