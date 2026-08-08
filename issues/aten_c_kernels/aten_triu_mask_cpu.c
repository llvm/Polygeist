#define M 32
#define N 32
void aten_triu_mask_cpu(int mask[M][N],int diagonal){for(int i=0;i<M;++i)for(int j=0;j<N;++j)mask[i][j]=j-i>=diagonal;}
