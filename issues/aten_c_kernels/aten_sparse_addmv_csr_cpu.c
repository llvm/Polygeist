#define R 64
#define C 64
#define N 512
void aten_sparse_addmv_csr_cpu(int ptr[R+1],int col[N],float val[N],float x[C],float out[R]){for(int r=0;r<R;++r){float v=0;for(int p=ptr[r];p<ptr[r+1];++p)v+=val[p]*x[col[p]];out[r]=v;}}
