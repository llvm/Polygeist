#define T 32
#define B 8
#define I 16
#define O 24
#define K 3
void aten_conv_tbc_cpu(float x[T][B][I],float w[K][I][O],float out[T-K+1][B][O]){for(int t=0;t<T-K+1;++t)for(int b=0;b<B;++b)for(int o=0;o<O;++o){float v=0;for(int k=0;k<K;++k)for(int i=0;i<I;++i)v+=x[t+k][b][i]*w[k][i][o];out[t][b][o]=v;}}
