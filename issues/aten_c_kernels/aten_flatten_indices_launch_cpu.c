#define D 3
#define N 512
void aten_flatten_indices_launch_cpu(int idx[D][N],int size[D],int out[N]){for(int n=0;n<N;++n){int v=0;for(int d=0;d<D;++d)v=v*size[d]+idx[d][n];out[n]=v;}}
