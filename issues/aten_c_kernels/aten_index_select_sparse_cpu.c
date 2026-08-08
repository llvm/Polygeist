#define N 512
#define K 128
void aten_index_select_sparse_cpu(float val[N],int idx[K],float out[K]){for(int k=0;k<K;++k)out[k]=val[idx[k]];}
