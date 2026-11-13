#define R 16
#define K 32
#define C 24
#define NNZ 96
void aten_sampled_addmm_sparse_csr_cpu(int crow[R+1],int col[NNZ],float self[NNZ],float a[R][K],float b[K][C],float alpha,float beta,float out[NNZ]){for(int r=0;r<R;++r)for(int p=crow[r];p<crow[r+1];++p){float s=0;for(int k=0;k<K;++k)s+=a[r][k]*b[k][col[p]];out[p]=beta*self[p]+alpha*s;}}
