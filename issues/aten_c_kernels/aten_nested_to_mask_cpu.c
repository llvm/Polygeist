#define B 8
#define N 64
void aten_nested_to_mask_cpu(int len[B],int out[B][N]){for(int b=0;b<B;++b)for(int i=0;i<N;++i)out[b][i]=i<len[b];}
