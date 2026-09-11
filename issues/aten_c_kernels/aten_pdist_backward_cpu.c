#ifndef N
#define N 16
#endif
#ifndef M
#define M 12
#endif
#ifndef D
#define D 32
#endif
extern float sqrtf(float);
void aten_pdist_backward_cpu(float x[N][D],float grad[N*(N-1)/2],float out[N][D]){float dist[N*(N-1)/2];for(int i=0;i<N;++i)for(int j=i+1;j<N;++j){int p=i*(2*N-i-1)/2+(j-i-1);dist[p]=0;for(int d=0;d<D;++d){float z=x[i][d]-x[j][d];dist[p]+=z*z;}dist[p]=sqrtf(dist[p]);}for(int i=0;i<N;++i)for(int d=0;d<D;++d)out[i][d]=0;for(int i=0;i<N;++i)for(int j=i+1;j<N;++j){int p=i*(2*N-i-1)/2+(j-i-1);float g=dist[p]==0?0:grad[p]/dist[p];for(int d=0;d<D;++d){float v=g*(x[i][d]-x[j][d]);out[i][d]+=v;out[j][d]-=v;}}}
