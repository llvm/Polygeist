#define N 512
#define Q 128
void aten_binary_search_strided_rightmost_cpu(int x[N],int q[Q],int out[Q]){for(int i=0;i<Q;++i){int l=0,r=N;while(l<r){int m=(l+r)/2;if(x[m]<=q[i])l=m+1;else r=m;}out[i]=l-1;}}
