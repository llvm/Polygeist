#define R 64
#define K 8
extern float expf(float);void aten_sparse_coo_softmax_cpu(float x[R][K],float out[R][K]){for(int r=0;r<R;++r){float s=0;for(int k=0;k<K;++k){out[r][k]=expf(x[r][k]);s+=out[r][k];}for(int k=0;k<K;++k)out[r][k]/=s;}}
