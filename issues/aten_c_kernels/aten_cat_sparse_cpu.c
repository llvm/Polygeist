#define B 4
#define N 256
void aten_cat_sparse_cpu(int idx[B][N],float val[B][N],int out_idx[B*N],float out_val[B*N]){for(int b=0;b<B;++b)for(int i=0;i<N;++i){out_idx[b*N+i]=idx[b][i];out_val[b*N+i]=val[b][i];}}
