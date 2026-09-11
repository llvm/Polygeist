#define N 128
void aten_gradient_float_cpu(float x[N],float coord[N],float out[N]){out[0]=(x[1]-x[0])/(coord[1]-coord[0]);for(int i=1;i<N-1;++i)out[i]=(x[i+1]-x[i-1])/(coord[i+1]-coord[i-1]);out[N-1]=(x[N-1]-x[N-2])/(coord[N-1]-coord[N-2]);}
