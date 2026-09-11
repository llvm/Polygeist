#define R 64
#define C 64
#define N 512
void aten_sparse_csr_reduce_dim0_cpu(int ptr[R+1],int col[N],float val[N],float out[C]){for(int c=0;c<C;++c)out[c]=0;for(int r=0;r<R;++r)for(int p=ptr[r];p<ptr[r+1];++p)out[col[p]]+=val[p];}
