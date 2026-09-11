#define N 32
#define P 3
void aten_circular_pad_cpu(float x[N],float out[N+2*P]){for(int i=0;i<N+2*P;++i){int j=(i-P)%N;if(j<0)j+=N;out[i]=x[j];}}
