#define R 64
#define C 64
#define N 512
void aten_dense_sparse_add_cpu(float dense[R][C],int row[N],int col[N],float value[N],float out[R][C]){for(int r=0;r<R;++r)for(int c=0;c<C;++c)out[r][c]=dense[r][c];for(int i=0;i<N;++i)out[row[i]][col[i]]+=value[i];}
