#ifndef N
#define N 4096
#endif
void aten_bernoulli_scalar_cpu(float uniform[N],float probability,float out[N]){for(int i=0;i<N;++i)out[i]=(float)(uniform[i]<probability);}
