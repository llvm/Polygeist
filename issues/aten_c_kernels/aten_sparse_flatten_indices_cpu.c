#define N 512
#define D 3
void aten_sparse_flatten_indices_cpu(int idx[D][N],int size[D],int out[N]){for(int n=0;n<N;++n){int v=0;for(int d=0;d<D;++d)v=v*size[d]+idx[d][n];out[n]=v;}}
