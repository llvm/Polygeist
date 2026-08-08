#define N 1024
void aten_sparse_add_values_cpu(float a[N],float b[N],float alpha,float out[N]){for(int i=0;i<N;++i)out[i]=a[i]+alpha*b[i];}
