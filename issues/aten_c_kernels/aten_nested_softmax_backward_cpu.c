#define B 8
#define N 64
void aten_nested_softmax_backward_cpu(float grad[B][N],float y[B][N],float out[B][N]){for(int b=0;b<B;++b){float s=0;for(int i=0;i<N;++i)s+=grad[b][i]*y[b][i];for(int i=0;i<N;++i)out[b][i]=y[b][i]*(grad[b][i]-s);}}
