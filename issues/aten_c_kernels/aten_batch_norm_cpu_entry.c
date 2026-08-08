#define N 2048
void aten_batch_norm_cpu_entry(float x[N],float scale,float bias,float out[N]){for(int i=0;i<N;++i)out[i]=x[i]*scale+bias;}
