#define R 32
#define K 64
extern float expf(float);void aten_host_softmax_cpu(float x[R][K],float out[R][K]){for(int r=0;r<R;++r){float m=x[r][0];for(int k=1;k<K;++k)m=x[r][k]>m?x[r][k]:m;float s=0;for(int k=0;k<K;++k){out[r][k]=expf(x[r][k]-m);s+=out[r][k];}for(int k=0;k<K;++k)out[r][k]/=s;}}
