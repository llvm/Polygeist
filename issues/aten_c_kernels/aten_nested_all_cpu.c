#ifndef B
#define B 8
#endif
#ifndef N
#define N 64
#endif
void aten_nested_all_cpu(int x[B][N],int len[B],int out[B]){for(int b=0;b<B;++b){int v=1;for(int i=0;i<len[b];++i)v&=x[b][i]!=0;out[b]=v;}}
