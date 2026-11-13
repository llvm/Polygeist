#ifndef N
#define N 4096
#endif
void aten_normal_cpu(float standard_normal[N],float mean,float std,float out[N]){for(int i=0;i<N;++i)out[i]=mean+std*standard_normal[i];}
