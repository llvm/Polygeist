#define N 4096
void aten_quant_saturation_cpu(int x[N],signed char out[N]){for(int i=0;i<N;++i){int v=x[i];if(v<-128)v=-128;if(v>127)v=127;out[i]=(signed char)v;}}
