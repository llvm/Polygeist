#ifndef N
#define N 4096
#endif
void aten_uniform_cpu(float uniform01[N],float from,float to,float out[N]){for(int i=0;i<N;++i)out[i]=from+(to-from)*uniform01[i];}
