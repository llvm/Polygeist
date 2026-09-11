#define R 64
#define K 64
#define C 48
#define N 512
void aten_sparse_csr_addmm_cpu(int ptr[R+1],int col[N],float val[N],float b[K][C],float out[R][C]){for(int r=0;r<R;++r)for(int c=0;c<C;++c){float v=0;for(int p=ptr[r];p<ptr[r+1];++p)v+=val[p]*b[col[p]][c];out[r][c]=v;}}
