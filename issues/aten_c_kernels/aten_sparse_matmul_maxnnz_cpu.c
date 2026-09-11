#define R 64
#define N 512
void aten_sparse_matmul_maxnnz_cpu(int ap[R+1],int ac[N],int bp[R+1],int bc[N],int out[R]){for(int r=0;r<R;++r){int n=0;for(int p=ap[r];p<ap[r+1];++p)n+=bp[ac[p]+1]-bp[ac[p]];out[r]=n;}}
