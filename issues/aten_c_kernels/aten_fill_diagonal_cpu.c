#ifndef N
#define N 32
#endif
void aten_fill_diagonal_cpu(float x[N][N],float value){for(int i=0;i<N;++i)x[i][i]=value;}
