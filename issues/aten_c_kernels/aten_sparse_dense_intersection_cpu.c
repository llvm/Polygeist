#define R 64
#define C 64
#define N 512
void aten_sparse_dense_intersection_cpu(float dense[R][C],int row[N],int col[N],float value[N],float out[N]){for(int i=0;i<N;++i)out[i]=value[i]*dense[row[i]][col[i]];}
