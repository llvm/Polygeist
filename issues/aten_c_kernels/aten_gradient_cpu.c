#define N 128
void aten_gradient_cpu(float x[N],float h,float out[N]){out[0]=(x[1]-x[0])/h;for(int i=1;i<N-1;++i)out[i]=(x[i+1]-x[i-1])/(2*h);out[N-1]=(x[N-1]-x[N-2])/h;}
