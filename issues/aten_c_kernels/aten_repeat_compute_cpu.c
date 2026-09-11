#define N 64
#define R 4
void aten_repeat_compute_cpu(float x[N],float out[R][N]){for(int r=0;r<R;++r)for(int i=0;i<N;++i)out[r][i]=x[i];}
