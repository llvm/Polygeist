#define N 32
#define K (N*(N+1)/2)
void aten_triu_indices_cpu(int row[K],int col[K]){int p=0;for(int i=0;i<N;++i)for(int j=i;j<N;++j){row[p]=i;col[p++]=j;}}
