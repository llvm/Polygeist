#define N 128
void aten_diff_cpu(float x[N],float out[N-1]){for(int i=0;i<N-1;++i)out[i]=x[i+1]-x[i];}
