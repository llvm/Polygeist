#define R 32
#define C 64
#define K 16
void aten_index_select_out_cpu(float x[R][C],int idx[K],float out[K][C]){for(int k=0;k<K;++k)for(int c=0;c<C;++c)out[k][c]=x[idx[k]][c];}
