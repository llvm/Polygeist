#define R 64
#define K 64
#define C 48
#define N 512
void aten_sspaddmm_cpu(int row[N],int col[N],float val[N],float b[K][C],float out[R][C]){for(int r=0;r<R;++r)for(int c=0;c<C;++c)out[r][c]=0;for(int p=0;p<N;++p)for(int c=0;c<C;++c)out[row[p]][c]+=val[p]*b[col[p]][c];}
