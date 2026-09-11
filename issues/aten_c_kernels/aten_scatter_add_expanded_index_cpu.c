#ifndef R
#define R 32
#endif
#ifndef K
#define K 64
#endif
#ifndef S
#define S 128
#endif
void aten_scatter_add_expanded_index_cpu(float out[R][S],int index[R][K],float source[R][K]){for(int r=0;r<R;++r)for(int k=0;k<K;++k)out[r][index[r][k]]+=source[r][k];}
