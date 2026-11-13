#ifndef N
#define N 4096
#endif
extern float logf(float);extern float ceilf(float);void aten_geometric_cpu(float uniform[N],float probability,float out[N]){float d=logf(1.0f-probability);for(int i=0;i<N;++i)out[i]=ceilf(logf(1.0f-uniform[i])/d);}
