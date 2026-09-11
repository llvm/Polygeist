#define N 32
#define K (N*(N+1)/2)
void aten_tril_indices_cpu(int row[K],int col[K]){for(int i=0;i<N;++i)for(int j=0;j<=i;++j){int p=i*(i+1)/2+j;row[p]=i;col[p]=j;}}
