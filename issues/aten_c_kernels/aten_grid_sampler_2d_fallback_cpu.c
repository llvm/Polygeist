#ifndef N
#define N 256
#endif
void aten_grid_sampler_2d_fallback_cpu(float input[N],int index[N],float out[N]){for(int i=0;i<N;++i)out[i]=index[i]>=0?input[index[i]]:0.0f;}
