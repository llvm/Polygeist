#define B 4
#define M 32
#define N 24
void aten_triu_tril_batch_cpu(float x[B][M][N],int diagonal,int upper,float out[B][M][N]){for(int b=0;b<B;++b)for(int i=0;i<M;++i)for(int j=0;j<N;++j)out[b][i][j]=(upper?(j-i>=diagonal):(j-i<=diagonal))?x[b][i][j]:0;}
