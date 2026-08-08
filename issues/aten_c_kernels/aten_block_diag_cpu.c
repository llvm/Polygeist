#define B 4
#define N 16
void aten_block_diag_cpu(float x[B][N][N],float out[B*N][B*N]){for(int i=0;i<B*N;++i)for(int j=0;j<B*N;++j)out[i][j]=0;for(int b=0;b<B;++b)for(int i=0;i<N;++i)for(int j=0;j<N;++j)out[b*N+i][b*N+j]=x[b][i][j];}
