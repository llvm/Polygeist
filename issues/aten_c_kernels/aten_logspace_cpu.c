#ifndef N
#define N 256
#endif
extern float powf(float,float);void aten_logspace_cpu(float start,float end,float base,float out[N]){for(int i=0;i<N;++i)out[i]=powf(base,start+(end-start)*i/(N-1));}
