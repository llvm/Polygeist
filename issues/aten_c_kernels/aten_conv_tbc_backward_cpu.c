#define T 30
#define B 8
#define I 16
#define O 24
#define K 3
void aten_conv_tbc_backward_cpu(float g[T][B][O],float w[K][I][O],float out[T+K-1][B][I]){for(int p=0;p<(T+K-1)*B*I;++p)((float*)out)[p]=0;for(int t=0;t<T;++t)for(int b=0;b<B;++b)for(int o=0;o<O;++o)for(int k=0;k<K;++k)for(int i=0;i<I;++i)out[t+k][b][i]+=g[t][b][o]*w[k][i][o];}
