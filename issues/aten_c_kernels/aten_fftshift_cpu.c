#define N 256
void aten_fftshift_cpu(float x[N],float out[N]){for(int i=0;i<N;++i)out[i]=x[(i+N/2)%N];}
