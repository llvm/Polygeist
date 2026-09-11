#ifndef R
#define R 32
#endif
#ifndef K
#define K 64
#endif
#ifndef S
#define S 128
#endif
void aten_index_fill_cpu(float out[S][K],int index[R],float value){for(int r=0;r<R;++r)for(int k=0;k<K;++k)out[index[r]][k]=value;}
