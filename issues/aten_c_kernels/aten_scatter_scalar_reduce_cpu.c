#ifndef R
#define R 32
#endif
#ifndef K
#define K 64
#endif
#ifndef S
#define S 128
#endif
void aten_scatter_scalar_reduce_cpu(float out[R][S],int index[R][K],float value,int reduce){for(int r=0;r<R;++r)for(int k=0;k<K;++k){int j=index[r][k];if(reduce==0)out[r][j]+=value;else if(reduce==1)out[r][j]*=value;else if(reduce==2)out[r][j]=out[r][j]>value?out[r][j]:value;else out[r][j]=out[r][j]<value?out[r][j]:value;}}
