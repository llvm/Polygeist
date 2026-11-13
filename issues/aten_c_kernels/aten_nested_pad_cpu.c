#define B 8
#define N 64
#define P 80
void aten_nested_pad_cpu(float x[B][N],float out[B][P]){for(int b=0;b<B;++b)for(int i=0;i<P;++i)out[b][i]=i<N?x[b][i]:0;}
