#define N 512
void aten_sparse_intersection_apply_cpu(float a[N],float b[N],float out[N]){for(int i=0;i<N;++i)out[i]=a[i]*b[i];}
