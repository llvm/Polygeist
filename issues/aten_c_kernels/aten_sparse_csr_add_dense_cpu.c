#define R 64
#define C 64
#define N 512
void aten_sparse_csr_add_dense_cpu(float out[R][C],int ptr[R+1],int col[N],float val[N]){for(int r=0;r<R;++r)for(int p=ptr[r];p<ptr[r+1];++p)out[r][col[p]]+=val[p];}
