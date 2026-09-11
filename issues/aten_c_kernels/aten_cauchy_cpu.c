#ifndef N
#define N 4096
#endif
extern float tanf(float);void aten_cauchy_cpu(float uniform[N],float median,float sigma,float out[N]){for(int i=0;i<N;++i)out[i]=median+sigma*tanf(3.14159265358979323846f*(uniform[i]-0.5f));}
