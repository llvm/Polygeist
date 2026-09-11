#define M 32
#define N 24
void aten_triu_tril_single_cpu(float x[M][N],int diagonal,int upper,float out[M][N]){for(int i=0;i<M;++i)for(int j=0;j<N;++j)out[i][j]=(upper?(j-i>=diagonal):(j-i<=diagonal))?x[i][j]:0;}
