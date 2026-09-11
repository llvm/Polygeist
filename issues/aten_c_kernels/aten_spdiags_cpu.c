#define D 5
#define N 16
void aten_spdiags_cpu(float diagonals[D][N],int offsets[D],float out[N][N]){for(int i=0;i<N;++i)for(int j=0;j<N;++j)out[i][j]=0;for(int d=0;d<D;++d)for(int i=0;i<N;++i){int j=i+offsets[d];if(j>=0&&j<N)out[i][j]=diagonals[d][offsets[d]>=0?i:i-offsets[d]];}}
