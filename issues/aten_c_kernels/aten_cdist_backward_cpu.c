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
void aten_cdist_backward_cpu(float x[N][D],float y[M][D],float grad[N][M],float out[N][D]){for(int i=0;i<N;++i)for(int d=0;d<D;++d)out[i][d]=0;for(int i=0;i<N;++i)for(int j=0;j<M;++j){float s=0;for(int d=0;d<D;++d){float z=x[i][d]-y[j][d];s+=z*z;}float n=sqrtf(s),g=n==0?0:grad[i][j]/n;for(int d=0;d<D;++d)out[i][d]+=g*(x[i][d]-y[j][d]);}}
