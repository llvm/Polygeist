#define R 32
#define C 64
#define K 16
void aten_index_select_dim1_cpu(float x[R][C],int idx[K],float out[R][K]){for(int r=0;r<R;++r)for(int k=0;k<K;++k)out[r][k]=x[r][idx[k]];}
