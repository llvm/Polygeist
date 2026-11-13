#define N 512
void aten_sparse_csr_reduce_all_cpu(float val[N],float out[1]){float v=0;for(int p=0;p<N;++p)v+=val[p];out[0]=v;}
