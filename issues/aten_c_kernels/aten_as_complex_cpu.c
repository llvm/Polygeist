#define N 512
void aten_as_complex_cpu(float x[N][2],float re[N],float im[N]){for(int i=0;i<N;++i){re[i]=x[i][0];im[i]=x[i][1];}}
