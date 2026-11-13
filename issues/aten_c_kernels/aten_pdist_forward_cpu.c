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
void aten_pdist_forward_cpu(float x[N][D],float out[N*(N-1)/2]){for(int i=0;i<N;++i)for(int j=i+1;j<N;++j){int p=i*(2*N-i-1)/2+(j-i-1);out[p]=0;for(int d=0;d<D;++d){float z=x[i][d]-x[j][d];out[p]+=z*z;}}for(int p=0;p<N*(N-1)/2;++p)out[p]=sqrtf(out[p]);}
