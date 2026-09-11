#define N 512
void aten_unsafe_index_cpu(float x[N],int idx[N],float out[N]){for(int i=0;i<N;++i)out[i]=x[idx[i]];}
