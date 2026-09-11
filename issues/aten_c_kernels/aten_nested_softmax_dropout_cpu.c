#define B 8
#define N 64
extern float expf(float);void aten_nested_softmax_dropout_cpu(float x[B][N],float mask[B][N],float out[B][N]){for(int b=0;b<B;++b){float s=0;for(int i=0;i<N;++i){out[b][i]=expf(x[b][i])*mask[b][i];s+=out[b][i];}for(int i=0;i<N;++i)out[b][i]/=s;}}
