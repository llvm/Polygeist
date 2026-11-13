#ifndef N
#define N 4096
#endif
#ifndef B0
#define B0 16
#endif
#ifndef B1
#define B1 12
#endif
void aten_histogramdd_cpu(float x[N][2],float weight[N],float lo0,float hi0,float lo1,float hi1,float out[B0][B1]){for(int a=0;a<B0;++a)for(int b=0;b<B1;++b)out[a][b]=0;for(int i=0;i<N;++i){int a=(int)((x[i][0]-lo0)*B0/(hi0-lo0));int b=(int)((x[i][1]-lo1)*B1/(hi1-lo1));if(a>=0&&a<B0&&b>=0&&b<B1)out[a][b]+=weight[i];}}
