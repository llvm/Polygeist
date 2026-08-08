#define N 256
#define M 128
void aten_lower_bound_cpu(float x[N],float v[M],int out[M]){for(int q=0;q<M;++q){int l=0,r=N;while(l<r){int m=(l+r)/2;if(x[m]<v[q])l=m+1;else r=m;}out[q]=l;}}
