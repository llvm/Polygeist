#define N 64
void aten_reflect_conj_tri_cpu(float re[N][N],float im[N][N]){for(int i=0;i<N;++i)for(int j=i+1;j<N;++j){re[j][i]=re[i][j];im[j][i]=-im[i][j];}}
