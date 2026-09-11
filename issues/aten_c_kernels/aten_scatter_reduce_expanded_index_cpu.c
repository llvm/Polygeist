#ifndef R
#define R 32
#endif
#ifndef K
#define K 64
#endif
#ifndef S
#define S 128
#endif
void aten_scatter_reduce_expanded_index_cpu(float out[R][S],int index[R][K],float source[R][K]){for(int r=0;r<R;++r)for(int k=0;k<K;++k){int j=index[r][k];out[r][j]=out[r][j]>source[r][k]?out[r][j]:source[r][k];}}
