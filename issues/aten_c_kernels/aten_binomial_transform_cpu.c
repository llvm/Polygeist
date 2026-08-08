#define N 1024
#define T 32
void aten_binomial_transform_cpu(int count[N],float p[N],float uniform[N][T],int out[N]){for(int i=0;i<N;++i){int k=0;for(int t=0;t<T&&t<count[i];++t)k+=uniform[i][t]<p[i];out[i]=k;}}
