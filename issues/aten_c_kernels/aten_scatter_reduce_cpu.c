#ifndef R
#define R 32
#endif
#ifndef K
#define K 64
#endif
#ifndef S
#define S 128
#endif
void aten_scatter_reduce_cpu(float out[R][S],int index[R][K],float source[R][K],int reduce){for(int r=0;r<R;++r)for(int k=0;k<K;++k){int j=index[r][k];float x=source[r][k];if(reduce==0)out[r][j]+=x;else if(reduce==1)out[r][j]*=x;else if(reduce==2)out[r][j]=out[r][j]>x?out[r][j]:x;else out[r][j]=out[r][j]<x?out[r][j]:x;}}
