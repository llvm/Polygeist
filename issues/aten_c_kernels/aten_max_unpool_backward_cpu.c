#define N 512
#define O 2048
void aten_max_unpool_backward_cpu(float grad[O],int index[N],float out[N]){for(int i=0;i<N;++i)out[i]=grad[index[i]];}
