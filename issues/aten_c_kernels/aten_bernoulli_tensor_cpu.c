#ifndef N
#define N 4096
#endif
void aten_bernoulli_tensor_cpu(float uniform[N],float probability[N],float out[N]){for(int i=0;i<N;++i)out[i]=(float)(uniform[i]<probability[i]);}
