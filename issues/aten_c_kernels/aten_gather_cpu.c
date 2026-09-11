#ifndef R
#define R 32
#endif
#ifndef K
#define K 64
#endif
#ifndef S
#define S 128
#endif
void aten_gather_cpu(float input[R][S],int index[R][K],float out[R][K]){for(int r=0;r<R;++r)for(int k=0;k<K;++k)out[r][k]=input[r][index[r][k]];}
