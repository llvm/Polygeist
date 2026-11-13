#define N 32
#define P 3
void aten_constant_pad_nd_cpu(float x[N],float value,float out[N+2*P]){for(int i=0;i<N+2*P;++i)out[i]=value;for(int i=0;i<N;++i)out[i+P]=x[i];}
