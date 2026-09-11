#ifndef N
#define N 64
#endif
void aten_trace_cpu(float x[N][N],float out[1]){float v=0;for(int i=0;i<N;++i)v+=x[i][i];out[0]=v;}
