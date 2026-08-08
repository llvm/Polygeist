#define N 512
#define O 64
void aten_index_reduce_impl_cpu(float x[N],int idx[N],float out[O]){for(int o=0;o<O;++o)out[o]=0;for(int i=0;i<N;++i)out[idx[i]]+=x[i];}
