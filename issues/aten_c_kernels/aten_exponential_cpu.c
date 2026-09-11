#ifndef N
#define N 4096
#endif
extern float log1pf(float);void aten_exponential_cpu(float uniform[N],float lambda,float out[N]){for(int i=0;i<N;++i)out[i]=-log1pf(-uniform[i])/lambda;}
