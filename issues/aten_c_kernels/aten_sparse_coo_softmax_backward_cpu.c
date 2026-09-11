#define R 64
#define K 8
void aten_sparse_coo_softmax_backward_cpu(float grad[R][K],float y[R][K],float out[R][K]){for(int r=0;r<R;++r){float s=0;for(int k=0;k<K;++k)s+=grad[r][k]*y[r][k];for(int k=0;k<K;++k)out[r][k]=y[r][k]*(grad[r][k]-s);}}
