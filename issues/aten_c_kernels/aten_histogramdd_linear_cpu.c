#ifndef N
#define N 4096
#endif
#ifndef B0
#define B0 16
#endif
#ifndef B1
#define B1 12
#endif
void aten_histogramdd_linear_cpu(int bin[N],float weight[N],float out[B0*B1]){for(int b=0;b<B0*B1;++b)out[b]=0;for(int i=0;i<N;++i)if(bin[i]>=0&&bin[i]<B0*B1)out[bin[i]]+=weight[i];}
