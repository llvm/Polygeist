#ifndef R
#define R 32
#endif
#ifndef K
#define K 64
#endif
#ifndef S
#define S 128
#endif
void aten_masked_scatter_cpu(float out[S],int mask[S],float source[S]){int p=0;for(int i=0;i<S;++i)if(mask[i])out[i]=source[p++];}
