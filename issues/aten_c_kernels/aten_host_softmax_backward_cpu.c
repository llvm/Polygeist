#define R 32
#define K 64
void aten_host_softmax_backward_cpu(float grad[R][K],float output[R][K],float out[R][K]){for(int r=0;r<R;++r){float s=0;for(int k=0;k<K;++k)s+=grad[r][k]*output[r][k];for(int k=0;k<K;++k)out[r][k]=output[r][k]*(grad[r][k]-s);}}
