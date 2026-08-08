#define B 8
#define N 64
void aten_nested_sum_backward_cpu(float grad[B],float out[B][N]){for(int b=0;b<B;++b)for(int i=0;i<N;++i)out[b][i]=grad[b];}
