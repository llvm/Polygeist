#define B 8
#define N 64
void aten_nested_sum_dim_cpu(float x[B][N],int len[B],float out[B]){for(int b=0;b<B;++b){float v=0;for(int i=0;i<len[b];++i)v+=x[b][i];out[b]=v;}}
