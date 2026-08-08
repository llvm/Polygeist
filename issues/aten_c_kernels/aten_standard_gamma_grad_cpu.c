#define N 1024
extern float logf(float);void aten_standard_gamma_grad_cpu(float a[N],float x[N],float out[N]){for(int i=0;i<N;++i)out[i]=(x[i]-a[i])/(a[i]+.001f)+logf(x[i]+.001f);}
