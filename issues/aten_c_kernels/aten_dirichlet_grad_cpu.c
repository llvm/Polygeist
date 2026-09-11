#define N 1024
extern float logf(float);void aten_dirichlet_grad_cpu(float x[N],float alpha[N],float total[N],float out[N]){for(int i=0;i<N;++i)out[i]=x[i]*(logf(x[i]+.001f)-alpha[i]/total[i]);}
