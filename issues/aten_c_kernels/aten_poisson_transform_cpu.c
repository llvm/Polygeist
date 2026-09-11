#define N 1024
#define T 64
extern float expf(float);void aten_poisson_transform_cpu(float lambda[N],float uniform[N][T],int out[N]){for(int i=0;i<N;++i){float q=1;int k=0;while(k<T&&q>expf(-lambda[i]))q*=uniform[i][k++];out[i]=k-1;}}
