#ifndef N
#define N 4096
#endif
extern float expf(float);void aten_log_normal_cpu(float standard_normal[N],float mean,float std,float out[N]){for(int i=0;i<N;++i)out[i]=expf(mean+std*standard_normal[i]);}
