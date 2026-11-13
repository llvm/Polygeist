#define R 64
#define N 512
void aten_sparse_csr_reduce_dim1_cpu(int ptr[R+1],float val[N],float out[R]){for(int r=0;r<R;++r){float v=0;for(int p=ptr[r];p<ptr[r+1];++p)v+=val[p];out[r]=v;}}
