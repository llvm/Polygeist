#ifndef R
#define R 32
#endif
#ifndef K
#define K 64
#endif
#ifndef S
#define S 128
#endif
void aten_flip_cpu(float input[R][K],float out[R][K],int flip_rows,int flip_cols){for(int r=0;r<R;++r)for(int k=0;k<K;++k){int rr=flip_rows?R-1-r:r;int kk=flip_cols?K-1-k:k;out[r][k]=input[rr][kk];}}
