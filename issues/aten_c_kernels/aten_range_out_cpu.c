#ifndef N
#define N 256
#endif
void aten_range_out_cpu(float start,float step,float out[N]){for(int i=0;i<N;++i)out[i]=start+i*step;}
