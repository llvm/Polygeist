#ifndef R
#define R 32
#endif
#ifndef K
#define K 64
#endif
#ifndef S
#define S 128
#endif
void aten_masked_fill_cpu(float out[S],int mask[S],float value){for(int i=0;i<S;++i)if(mask[i])out[i]=value;}
