#define N 1024
void aten_std_var_all_cpu(float x[N],float out[1]){float m=0;for(int i=0;i<N;++i)m+=x[i];m/=N;float v=0;for(int i=0;i<N;++i){float d=x[i]-m;v+=d*d;}out[0]=v/(N-1);}
