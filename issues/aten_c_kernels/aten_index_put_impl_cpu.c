#define N 512
void aten_index_put_impl_cpu(float out[N],int idx[N],float value[N]){for(int i=0;i<N;++i)out[idx[i]]=value[i];}
