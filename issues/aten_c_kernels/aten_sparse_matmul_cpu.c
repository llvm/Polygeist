#define R 64
#define C 64
#define N 512
void aten_sparse_matmul_cpu(int ap[R+1],int ac[N],float av[N],float b[R][C],float out[R][C]){for(int r=0;r<R;++r)for(int c=0;c<C;++c){float v=0;for(int p=ap[r];p<ap[r+1];++p)v+=av[p]*b[ac[p]][c];out[r][c]=v;}}
