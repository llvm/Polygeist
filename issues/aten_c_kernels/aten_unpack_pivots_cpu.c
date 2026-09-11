#define N 128
void aten_unpack_pivots_cpu(int piv[N],int perm[N]){for(int i=0;i<N;++i)perm[i]=i;for(int i=0;i<N;++i){int j=piv[i]-1,t=perm[i];perm[i]=perm[j];perm[j]=t;}}
