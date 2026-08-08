#define N 1024
extern float sqrtf(float);void aten_sparse_norm_cpu(float value[N],float out[1]){float v=0;for(int i=0;i<N;++i)v+=value[i]*value[i];out[0]=sqrtf(v);}
