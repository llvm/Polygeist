#define K 64
#define N 48
void aten_quant_col_offsets_cpu(signed char w[K][N],int zero,int out[N]){for(int n=0;n<N;++n){int v=0;for(int k=0;k<K;++k)v+=(int)w[k][n];out[n]=v-zero*K;}}
