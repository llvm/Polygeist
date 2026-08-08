#define B 4
#define N 64
void aten_copy_tensor_array_cpu(float x[B][N],float out[B][N]){for(int b=0;b<B;++b)for(int i=0;i<N;++i)out[b][i]=x[b][i];}
