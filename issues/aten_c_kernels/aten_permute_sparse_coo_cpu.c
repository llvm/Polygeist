#define D 3
#define N 512
void aten_permute_sparse_coo_cpu(int idx[D][N],int perm[D],int out[D][N]){for(int d=0;d<D;++d)for(int n=0;n<N;++n)out[d][n]=idx[perm[d]][n];}
